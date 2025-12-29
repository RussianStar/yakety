extern "C" {
#include "transcription.h"
#include "logging.h"
#include "utils.h"
#include "preferences.h"
#include "models.h"
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <thread>

#include "whisper.h"
#include "../whisper.cpp/ggml/include/ggml.h"

#ifdef YAKETY_HAVE_CURL
#include <curl/curl.h>
#endif
#ifdef YAKETY_HAVE_WINHTTP
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winhttp.h>
#endif

static const char *kDefaultRemoteEndpoint = "http://192.168.178.242:4141/voice/v1";
static const char *kDefaultRemoteModel = "voxtral";

static struct whisper_context *ctx = NULL;
static utils_mutex_t *ctx_mutex = NULL;  // Thread safety for transcription context
static char g_language[16] = "en"; // Default to English

static char *g_remote_endpoint = NULL;
static char *g_remote_model = NULL;

#ifdef YAKETY_HAVE_CURL
static bool g_curl_initialized = false;
#endif

typedef struct {
    TranscriptionProvider id;
    const char *name;
    bool available;
    int (*init)(const char *model_path);
    char *(*process)(const float *audio_data, int n_samples, int sample_rate);
    void (*cleanup)(void);
} TranscriptionProviderOps;

static const TranscriptionProviderOps *g_provider_ops = NULL;

// Initialize mutex on first use
static void ensure_mutex_initialized(void) {
    if (ctx_mutex == NULL) {
        ctx_mutex = utils_mutex_create();
    }
}

// Custom log callback that suppresses whisper/ggml logs
static void null_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void) level;
    (void) text;
    (void) user_data;
    // Do nothing - suppress all whisper/ggml logs
}

static const char *get_pref_string_or_default(const char *key, const char *fallback) {
    const char *value = preferences_get_string(key);
    if (value && value[0] != '\0') {
        return value;
    }
    return fallback;
}

static TranscriptionProvider parse_provider_pref(const char *value) {
    if (!value || value[0] == '\0') {
        return TRANSCRIPTION_PROVIDER_REMOTE_HTTP;
    }
    if (utils_stricmp(value, "whisper") == 0 || utils_stricmp(value, "local") == 0) {
        return TRANSCRIPTION_PROVIDER_WHISPER;
    }
    if (utils_stricmp(value, "remote") == 0 || utils_stricmp(value, "remote_http") == 0 ||
        utils_stricmp(value, "http") == 0) {
        return TRANSCRIPTION_PROVIDER_REMOTE_HTTP;
    }
    return TRANSCRIPTION_PROVIDER_REMOTE_HTTP;
}

TranscriptionProvider transcription_get_provider(void) {
    return parse_provider_pref(preferences_get_string("transcription_provider"));
}

const char *transcription_get_provider_name(void) {
    if (g_provider_ops) {
        return g_provider_ops->name;
    }
    switch (transcription_get_provider()) {
        case TRANSCRIPTION_PROVIDER_REMOTE_HTTP:
            return "remote_http";
        case TRANSCRIPTION_PROVIDER_WHISPER:
        default:
            return "whisper";
    }
}

static char *normalize_transcription_text(char *result) {
    if (!result) {
        return NULL;
    }

    // Trim whitespace
    char *start = result;
    while (*start && std::isspace(static_cast<unsigned char>(*start))) {
        start++;
    }

    char *end = result + strlen(result);
    while (end > start && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
        end--;
    }
    *end = '\0';

    if (start != result) {
        memmove(result, start, strlen(start) + 1);
    }

    // Filter out bracketed tokens (non-speech annotations)
    size_t len = strlen(result);
    if (len >= 2) {
        if ((result[0] == '[' || result[0] == '*') &&
            (result[len - 1] == ']' || result[len - 1] == '*')) {
            result[0] = '\0';
            return result;
        }
    }

    // Replace double spaces with single spaces
    char *read = result;
    char *write = result;
    bool prev_space = false;

    while (*read) {
        if (*read == ' ') {
            if (!prev_space) {
                *write++ = *read;
                prev_space = true;
            }
        } else {
            *write++ = *read;
            prev_space = false;
        }
        read++;
    }
    *write = '\0';

    // Add trailing space for convenient pasting (unless result is empty)
    len = strlen(result);
    if (len > 0) {
        char *expanded = (char *) realloc(result, len + 2);
        if (!expanded) {
            return result;
        }
        result = expanded;
        result[len] = ' ';
        result[len + 1] = '\0';
    }

    return result;
}

static char *dup_with_padding(const char *text) {
    size_t len = text ? strlen(text) : 0;
    char *copy = (char *) malloc(len + 2);
    if (!copy) {
        return NULL;
    }
    if (len > 0 && text) {
        memcpy(copy, text, len);
    }
    copy[len] = '\0';
    copy[len + 1] = '\0';
    return copy;
}

// --- Whisper provider ---

static int whisper_provider_init(const char *model_path) {
    const char *resolved_model_path = model_path;
    if (!resolved_model_path || resolved_model_path[0] == '\0') {
        resolved_model_path = utils_get_model_path();
    }

    if (!resolved_model_path) {
        log_error("ERROR: No model path provided");
        return -1;
    }

    if (ctx != NULL) {
        log_info("Transcription already initialized");
        return 0;
    }

    // Disable whisper/ggml logging
    ggml_log_set(null_log_callback, NULL);
    whisper_log_set(null_log_callback, NULL);

    log_info("🧠 Loading Whisper model: %s", resolved_model_path);

    double start = utils_now();

    struct whisper_context_params cparams = whisper_context_default_params();

    // Enable Flash Attention for better performance
    cparams.flash_attn = true;
    cparams.use_gpu = true; // Ensure GPU is enabled for Flash Attention

    // Log what we're requesting
    log_info("🔧 Requesting Flash Attention: %s, GPU: %s\n",
             cparams.flash_attn ? "YES" : "NO",
             cparams.use_gpu ? "YES" : "NO");

    ctx = whisper_init_from_file_with_params(resolved_model_path, cparams);

    double duration = utils_now() - start;

    if (!ctx) {
        log_error("ERROR: Failed to initialize Whisper from model file: %s", resolved_model_path);
        return -1;
    }

    log_info("✅ Whisper initialized successfully (took %.0f ms)", duration * 1000.0);
    log_info("⚡ Requested - Flash Attention: %s, GPU: %s",
             cparams.flash_attn ? "enabled" : "disabled",
             cparams.use_gpu ? "enabled" : "disabled");

    // Check and log VAD status during initialization
    bool vad_enabled = preferences_get_bool("vad_enabled", true);
    const char *vad_model_path = models_get_vad_path();
    if (vad_enabled && vad_model_path) {
        log_info("🎙️ VAD (Voice Activity Detection): ENABLED");
    } else if (!vad_enabled) {
        log_info("🎙️ VAD (Voice Activity Detection): DISABLED (set vad_enabled=true in config or use menu to enable)");
    } else {
        log_info("🎙️ VAD (Voice Activity Detection): DISABLED (model not found)");
    }

    return 0;
}

static char *whisper_provider_process(const float *audio_data, int n_samples, int sample_rate) {
    (void) sample_rate; // Currently unused

    if (ctx == NULL) {
        log_error("ERROR: Whisper not initialized");
        return NULL;
    }

    // Set up whisper parameters
    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.print_timestamps = false;
    wparams.print_special = false;
    wparams.translate = false;
    wparams.language = g_language; // Use configured language
    // Use optimal number of threads (leave some for system)
    int n_threads = std::thread::hardware_concurrency();
    if (n_threads > 1) {
        n_threads = std::min(n_threads - 1, 8); // Leave one core for system, cap at 8
    } else {
        n_threads = 4; // Default fallback
    }
    wparams.n_threads = n_threads;
    wparams.offset_ms = 0;
    wparams.duration_ms = 0;

    // Configure VAD (Voice Activity Detection)
    bool vad_enabled = preferences_get_bool("vad_enabled", true);
    const char *vad_model_path = models_get_vad_path();
    if (vad_enabled && vad_model_path) {
        wparams.vad = true;
        wparams.vad_model_path = vad_model_path;
        wparams.vad_params = whisper_vad_default_params();
        log_info("VAD enabled with model: %s", vad_model_path);
    } else {
        wparams.vad = false;
        if (!vad_enabled) {
            log_info("VAD disabled in preferences");
        } else {
            log_info("VAD model not found, running without voice activity detection");
        }
    }

    // Run transcription
    double whisper_start = utils_now();
    int whisper_result = whisper_full(ctx, wparams, audio_data, n_samples);
    double whisper_duration = utils_now() - whisper_start;

    log_info("⏱️  Whisper inference took: %.0f ms\n", whisper_duration * 1000.0);

    if (whisper_result != 0) {
        log_error("ERROR: Failed to run whisper transcription\n");
        return NULL;
    }

    // Get transcription result
    const int n_segments = whisper_full_n_segments(ctx);
    if (n_segments == 0) {
        char *empty_result = (char *) malloc(1);
        if (empty_result) {
            empty_result[0] = '\0';
        }
        return empty_result;
    }

    // Calculate total length needed
    size_t total_len = 0;
    for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        if (text) {
            total_len += strlen(text);
            if (i > 0) total_len++; // Space separator
        }
    }

    char *result = (char *) malloc(total_len + 2);
    if (!result) {
        log_error("ERROR: Failed to allocate memory for transcription\n");
        return NULL;
    }

    // Concatenate all segments
    result[0] = '\0';
    for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        if (text) {
            if (strlen(result) > 0) {
                strcat(result, " ");
            }
            strcat(result, text);
        }
    }

    return result;
}

static void whisper_provider_cleanup(void) {
    if (ctx != NULL) {
        struct whisper_context *old_ctx = ctx;
        ctx = NULL;  // Set to NULL first to prevent double cleanup
        if (old_ctx != NULL) {
            whisper_free(old_ctx);
        }
    }
}

// --- Remote HTTP provider ---

static void remote_provider_cleanup(void) {
    if (g_remote_endpoint) {
        free(g_remote_endpoint);
        g_remote_endpoint = NULL;
    }
    if (g_remote_model) {
        free(g_remote_model);
        g_remote_model = NULL;
    }
#ifdef YAKETY_HAVE_CURL
    if (g_curl_initialized) {
        curl_global_cleanup();
        g_curl_initialized = false;
    }
#endif
}

static std::string build_transcription_url(const char *endpoint) {
    if (!endpoint || endpoint[0] == '\0') {
        return std::string();
    }
    std::string url(endpoint);
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    if (url.find("/audio/transcriptions") != std::string::npos ||
        url.find("/transcriptions") != std::string::npos) {
        return url;
    }
    return url + "/audio/transcriptions";
}

static void write_u16_le(uint8_t *dst, uint16_t value) {
    dst[0] = (uint8_t) (value & 0xFF);
    dst[1] = (uint8_t) ((value >> 8) & 0xFF);
}

static void write_u32_le(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t) (value & 0xFF);
    dst[1] = (uint8_t) ((value >> 8) & 0xFF);
    dst[2] = (uint8_t) ((value >> 16) & 0xFF);
    dst[3] = (uint8_t) ((value >> 24) & 0xFF);
}

static bool encode_wav_pcm16(const float *audio_data, int n_samples, int sample_rate, std::vector<uint8_t> &out) {
    if (!audio_data || n_samples <= 0 || sample_rate <= 0) {
        return false;
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t bytes_per_sample = bits_per_sample / 8;
    const uint32_t data_size = (uint32_t) n_samples * channels * bytes_per_sample;
    const uint32_t riff_size = 36 + data_size;

    out.resize(44 + data_size);
    uint8_t *p = out.data();

    memcpy(p, "RIFF", 4);
    write_u32_le(p + 4, riff_size);
    memcpy(p + 8, "WAVE", 4);

    memcpy(p + 12, "fmt ", 4);
    write_u32_le(p + 16, 16); // PCM header size
    write_u16_le(p + 20, 1);  // PCM format
    write_u16_le(p + 22, channels);
    write_u32_le(p + 24, (uint32_t) sample_rate);
    write_u32_le(p + 28, (uint32_t) sample_rate * channels * bytes_per_sample);
    write_u16_le(p + 32, channels * bytes_per_sample);
    write_u16_le(p + 34, bits_per_sample);

    memcpy(p + 36, "data", 4);
    write_u32_le(p + 40, data_size);

    size_t offset = 44;
    for (int i = 0; i < n_samples; ++i) {
        float clamped = std::max(-1.0f, std::min(1.0f, audio_data[i]));
        int16_t sample = (int16_t) lrintf(clamped * 32767.0f);
        p[offset++] = (uint8_t) (sample & 0xFF);
        p[offset++] = (uint8_t) ((sample >> 8) & 0xFF);
    }

    return true;
}

static char *extract_json_text(const char *response) {
    if (!response) {
        return NULL;
    }

    const char *key = "\"text\"";
    const char *pos = strstr(response, key);
    if (!pos) {
        return NULL;
    }
    pos = strchr(pos + strlen(key), ':');
    if (!pos) {
        return NULL;
    }
    pos++;
    while (*pos && std::isspace(static_cast<unsigned char>(*pos))) {
        pos++;
    }
    if (*pos != '\"') {
        return NULL;
    }
    pos++;

    std::string out;
    while (*pos) {
        if (*pos == '\\') {
            pos++;
            if (!*pos) break;
            switch (*pos) {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case '\\': out.push_back('\\'); break;
                case '\"': out.push_back('\"'); break;
                case 'u': {
                    int value = 0;
                    bool ok = true;
                    for (int i = 0; i < 4; ++i) {
                        pos++;
                        if (!*pos) { ok = false; break; }
                        char c = *pos;
                        int digit = 0;
                        if (c >= '0' && c <= '9') digit = c - '0';
                        else if (c >= 'a' && c <= 'f') digit = 10 + (c - 'a');
                        else if (c >= 'A' && c <= 'F') digit = 10 + (c - 'A');
                        else { ok = false; break; }
                        value = (value << 4) | digit;
                    }
                    if (ok && value >= 0 && value <= 0x7F) {
                        out.push_back((char) value);
                    } else if (ok) {
                        out.push_back('?');
                    }
                    break;
                }
                default:
                    out.push_back(*pos);
                    break;
            }
        } else if (*pos == '\"') {
            break;
        } else {
            out.push_back(*pos);
        }
        pos++;
    }

    char *result = (char *) malloc(out.size() + 2);
    if (!result) {
        return NULL;
    }
    memcpy(result, out.c_str(), out.size());
    result[out.size()] = '\0';
    result[out.size() + 1] = '\0';
    return result;
}

static void append_bytes(std::vector<uint8_t> &out, const void *data, size_t size) {
    if (!data || size == 0) {
        return;
    }
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
}

static void append_string(std::vector<uint8_t> &out, const std::string &text) {
    append_bytes(out, text.data(), text.size());
}

static std::string make_boundary(void) {
    char buffer[96];
    unsigned long long stamp = (unsigned long long) (utils_now() * 1000000.0);
    snprintf(buffer, sizeof(buffer), "----yakety-boundary-%p-%llu", utils_thread_id(), stamp);
    return std::string(buffer);
}

static void append_form_field(std::vector<uint8_t> &out,
                              const std::string &boundary,
                              const char *name,
                              const char *value) {
    if (!name || !value) {
        return;
    }
    append_string(out, "--" + boundary + "\r\n");
    append_string(out, "Content-Disposition: form-data; name=\"");
    append_string(out, name);
    append_string(out, "\"\r\n\r\n");
    append_string(out, value);
    append_string(out, "\r\n");
}

static void append_file_field(std::vector<uint8_t> &out,
                              const std::string &boundary,
                              const char *name,
                              const char *filename,
                              const char *content_type,
                              const std::vector<uint8_t> &data) {
    append_string(out, "--" + boundary + "\r\n");
    append_string(out, "Content-Disposition: form-data; name=\"");
    append_string(out, name ? name : "file");
    append_string(out, "\"; filename=\"");
    append_string(out, filename ? filename : "audio.wav");
    append_string(out, "\"\r\n");
    append_string(out, "Content-Type: ");
    append_string(out, content_type ? content_type : "application/octet-stream");
    append_string(out, "\r\n\r\n");
    append_bytes(out, data.data(), data.size());
    append_string(out, "\r\n");
}

static std::vector<uint8_t> build_multipart_body(const std::string &boundary,
                                                 const std::vector<uint8_t> &wav_data,
                                                 const char *model,
                                                 const char *language) {
    std::vector<uint8_t> body;
    append_file_field(body, boundary, "file", "audio.wav", "audio/wav", wav_data);

    if (model && model[0] != '\0') {
        append_form_field(body, boundary, "model", model);
    }

    if (language && language[0] != '\0') {
        append_form_field(body, boundary, "language", language);
    }

    append_string(body, "--" + boundary + "--\r\n");
    return body;
}

#ifdef YAKETY_HAVE_CURL

typedef struct {
    char *data;
    size_t size;
} CurlBuffer;

static size_t curl_write_callback(char *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t total = size * nmemb;
    CurlBuffer *buffer = (CurlBuffer *) userdata;

    char *new_data = (char *) realloc(buffer->data, buffer->size + total + 1);
    if (!new_data) {
        return 0;
    }

    buffer->data = new_data;
    memcpy(buffer->data + buffer->size, ptr, total);
    buffer->size += total;
    buffer->data[buffer->size] = '\0';
    return total;
}

#endif

#ifdef YAKETY_HAVE_WINHTTP

static std::wstring utf8_to_wide_string(const char *utf8) {
    if (!utf8) {
        return std::wstring();
    }
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    if (len <= 1) {
        return std::wstring();
    }
    std::wstring wide;
    wide.resize(len - 1);
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, &wide[0], len);
    return wide;
}

static bool winhttp_crack_url(const std::string &url,
                              std::wstring &host,
                              INTERNET_PORT &port,
                              std::wstring &path,
                              bool &secure) {
    std::wstring wurl = utf8_to_wide_string(url.c_str());
    if (wurl.empty()) {
        return false;
    }

    URL_COMPONENTS components = {};
    components.dwStructSize = sizeof(components);
    components.dwSchemeLength = (DWORD) -1;
    components.dwHostNameLength = (DWORD) -1;
    components.dwUrlPathLength = (DWORD) -1;
    components.dwExtraInfoLength = (DWORD) -1;

    if (!WinHttpCrackUrl(wurl.c_str(), (DWORD) wurl.size(), 0, &components)) {
        return false;
    }

    if (!components.lpszHostName || components.dwHostNameLength == 0) {
        return false;
    }

    host.assign(components.lpszHostName, components.dwHostNameLength);
    path.clear();
    if (components.lpszUrlPath && components.dwUrlPathLength > 0) {
        path.append(components.lpszUrlPath, components.dwUrlPathLength);
    }
    if (components.lpszExtraInfo && components.dwExtraInfoLength > 0) {
        path.append(components.lpszExtraInfo, components.dwExtraInfoLength);
    }
    if (path.empty()) {
        path = L"/";
    }

    port = components.nPort;
    secure = (components.nScheme == INTERNET_SCHEME_HTTPS);
    return true;
}

static bool winhttp_post_multipart(const std::string &url,
                                   const std::vector<uint8_t> &body,
                                   const std::string &boundary,
                                   std::string &response,
                                   long &http_status) {
    std::wstring host;
    std::wstring path;
    INTERNET_PORT port = 0;
    bool secure = false;

    if (!winhttp_crack_url(url, host, port, path, secure)) {
        return false;
    }

    HINTERNET session = WinHttpOpen(L"yakety/1.0",
                                    WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                    WINHTTP_NO_PROXY_NAME,
                                    WINHTTP_NO_PROXY_BYPASS,
                                    0);
    if (!session) {
        return false;
    }

    WinHttpSetTimeouts(session, 10000, 10000, 30000, 120000);

    HINTERNET connect = WinHttpConnect(session, host.c_str(), port, 0);
    if (!connect) {
        WinHttpCloseHandle(session);
        return false;
    }

    DWORD flags = secure ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET request = WinHttpOpenRequest(connect, L"POST", path.c_str(), NULL,
                                           WINHTTP_NO_REFERER,
                                           WINHTTP_DEFAULT_ACCEPT_TYPES,
                                           flags);
    if (!request) {
        WinHttpCloseHandle(connect);
        WinHttpCloseHandle(session);
        return false;
    }

    std::wstring header = L"Content-Type: multipart/form-data; boundary=" + utf8_to_wide_string(boundary.c_str());
    WinHttpAddRequestHeaders(request, header.c_str(), (DWORD) -1, WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE);

    BOOL ok = WinHttpSendRequest(request,
                                 WINHTTP_NO_ADDITIONAL_HEADERS,
                                 0,
                                 body.empty() ? WINHTTP_NO_REQUEST_DATA : (LPVOID) body.data(),
                                 (DWORD) body.size(),
                                 (DWORD) body.size(),
                                 0);
    if (!ok) {
        WinHttpCloseHandle(request);
        WinHttpCloseHandle(connect);
        WinHttpCloseHandle(session);
        return false;
    }

    ok = WinHttpReceiveResponse(request, NULL);
    if (!ok) {
        WinHttpCloseHandle(request);
        WinHttpCloseHandle(connect);
        WinHttpCloseHandle(session);
        return false;
    }

    DWORD status_code = 0;
    DWORD status_len = sizeof(status_code);
    if (WinHttpQueryHeaders(request,
                            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX,
                            &status_code,
                            &status_len,
                            WINHTTP_NO_HEADER_INDEX)) {
        http_status = (long) status_code;
    } else {
        http_status = 0;
    }

    response.clear();
    bool success = true;
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request, &available)) {
            success = false;
            break;
        }
        if (available == 0) {
            break;
        }
        std::vector<char> buffer(available);
        DWORD read = 0;
        if (!WinHttpReadData(request, buffer.data(), available, &read)) {
            success = false;
            break;
        }
        if (read > 0) {
            response.append(buffer.data(), buffer.data() + read);
        }
    }

    WinHttpCloseHandle(request);
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);

    return success;
}

#endif

static int remote_provider_init(const char *unused) {
    (void) unused;
#if !defined(YAKETY_HAVE_CURL) && !defined(YAKETY_HAVE_WINHTTP)
    log_error("ERROR: Remote transcription provider unavailable (no HTTP backend linked)");
    return -1;
#else
#ifdef YAKETY_HAVE_CURL
    if (!g_curl_initialized) {
        CURLcode init_result = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (init_result != CURLE_OK) {
            log_error("ERROR: curl_global_init failed: %s", curl_easy_strerror(init_result));
            return -1;
        }
        g_curl_initialized = true;
    }
#endif

    const char *endpoint = get_pref_string_or_default("transcription_endpoint", kDefaultRemoteEndpoint);
    if (!endpoint || endpoint[0] == '\0') {
        log_error("ERROR: No transcription endpoint configured");
        return -1;
    }

    const char *model = get_pref_string_or_default("transcription_model", kDefaultRemoteModel);

    if (g_remote_endpoint) {
        free(g_remote_endpoint);
        g_remote_endpoint = NULL;
    }
    if (g_remote_model) {
        free(g_remote_model);
        g_remote_model = NULL;
    }

    g_remote_endpoint = utils_strdup(endpoint);
    g_remote_model = utils_strdup(model ? model : "");

    if (!g_remote_endpoint) {
        log_error("ERROR: Failed to allocate transcription endpoint string");
        return -1;
    }

    log_info("🌐 Using remote transcription endpoint: %s", g_remote_endpoint);
    if (g_remote_model && g_remote_model[0] != '\0') {
        log_info("🧠 Remote transcription model: %s", g_remote_model);
    }

    return 0;
#endif
}

static char *remote_provider_process(const float *audio_data, int n_samples, int sample_rate) {
#if defined(YAKETY_HAVE_CURL)
    if (!g_remote_endpoint || g_remote_endpoint[0] == '\0') {
        log_error("ERROR: No transcription endpoint configured");
        return NULL;
    }

    std::string url = build_transcription_url(g_remote_endpoint);
    if (url.empty()) {
        log_error("ERROR: Invalid transcription endpoint");
        return NULL;
    }

    std::vector<uint8_t> wav_data;
    if (!encode_wav_pcm16(audio_data, n_samples, sample_rate, wav_data)) {
        log_error("ERROR: Failed to encode audio for remote transcription");
        return NULL;
    }

    CURL *curl = curl_easy_init();
    if (!curl) {
        log_error("ERROR: Failed to initialize CURL handle");
        return NULL;
    }

    CurlBuffer response = {0};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "yakety/1.0");
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    curl_mime *mime = curl_mime_init(curl);
    curl_mimepart *part = curl_mime_addpart(mime);
    curl_mime_name(part, "file");
    curl_mime_filename(part, "audio.wav");
    curl_mime_type(part, "audio/wav");
    curl_mime_data(part, (const char *) wav_data.data(), wav_data.size());

    const char *model = g_remote_model && g_remote_model[0] != '\0' ? g_remote_model : kDefaultRemoteModel;
    if (model && model[0] != '\0') {
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "model");
        curl_mime_data(part, model, CURL_ZERO_TERMINATED);
    }

    if (g_language[0] != '\0' && utils_stricmp(g_language, "auto") != 0) {
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "language");
        curl_mime_data(part, g_language, CURL_ZERO_TERMINATED);
    }

    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_mime_free(mime);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        log_error("ERROR: Remote transcription request failed: %s", curl_easy_strerror(res));
        free(response.data);
        return NULL;
    }

    if (http_code < 200 || http_code >= 300) {
        log_error("ERROR: Remote transcription failed with HTTP %ld", http_code);
        if (response.data) {
            log_debug("Remote response: %s", response.data);
        }
        free(response.data);
        return NULL;
    }

    char *text = extract_json_text(response.data);
    if (!text) {
        if (response.data) {
            text = dup_with_padding(response.data);
        } else {
            text = dup_with_padding(\"\");
        }
    }

    free(response.data);
    return text;
#elif defined(YAKETY_HAVE_WINHTTP)
    if (!g_remote_endpoint || g_remote_endpoint[0] == '\0') {
        log_error("ERROR: No transcription endpoint configured");
        return NULL;
    }

    std::string url = build_transcription_url(g_remote_endpoint);
    if (url.empty()) {
        log_error("ERROR: Invalid transcription endpoint");
        return NULL;
    }

    std::vector<uint8_t> wav_data;
    if (!encode_wav_pcm16(audio_data, n_samples, sample_rate, wav_data)) {
        log_error("ERROR: Failed to encode audio for remote transcription");
        return NULL;
    }

    const char *model = g_remote_model && g_remote_model[0] != '\0' ? g_remote_model : kDefaultRemoteModel;
    const char *language = NULL;
    if (g_language[0] != '\0' && utils_stricmp(g_language, "auto") != 0) {
        language = g_language;
    }

    std::string boundary = make_boundary();
    std::vector<uint8_t> body = build_multipart_body(boundary, wav_data, model, language);

    std::string response;
    long http_code = 0;
    if (!winhttp_post_multipart(url, body, boundary, response, http_code)) {
        log_error("ERROR: Remote transcription request failed (WinHTTP)");
        return NULL;
    }

    if (http_code < 200 || http_code >= 300) {
        log_error("ERROR: Remote transcription failed with HTTP %ld", http_code);
        if (!response.empty()) {
            log_debug("Remote response: %s", response.c_str());
        }
        return NULL;
    }

    char *text = extract_json_text(response.c_str());
    if (!text) {
        text = dup_with_padding(response.c_str());
    }

    return text;
#else
    (void) audio_data;
    (void) n_samples;
    (void) sample_rate;
    log_error("ERROR: Remote transcription provider unavailable (no HTTP backend linked)");
    return NULL;
#endif
}

static const TranscriptionProviderOps kWhisperProvider = {
    TRANSCRIPTION_PROVIDER_WHISPER,
    "whisper",
    true,
    whisper_provider_init,
    whisper_provider_process,
    whisper_provider_cleanup,
};

static const TranscriptionProviderOps kRemoteProvider = {
    TRANSCRIPTION_PROVIDER_REMOTE_HTTP,
    "remote_http",
#if defined(YAKETY_HAVE_CURL) || defined(YAKETY_HAVE_WINHTTP)
    true,
#else
    false,
#endif
    remote_provider_init,
    remote_provider_process,
    remote_provider_cleanup,
};

void transcription_set_language(const char *language) {
    ensure_mutex_initialized();
    utils_mutex_lock(ctx_mutex);

    if (language && strlen(language) > 0) {
        strncpy(g_language, language, sizeof(g_language) - 1);
        g_language[sizeof(g_language) - 1] = '\0';
        log_info("🌐 Transcription language set to: %s\n", g_language);
    }

    utils_mutex_unlock(ctx_mutex);
}

int transcription_init(const char *model_path) {
    ensure_mutex_initialized();

    log_debug("transcription_init() ENTRY - thread=%p, model_path=%s",
              utils_thread_id(), model_path ? model_path : "NULL");

    utils_mutex_lock(ctx_mutex);
    log_debug("Acquired transcription mutex - thread=%p", utils_thread_id());

    // Check if already initialized
    if (g_provider_ops != NULL) {
        log_debug("Already initialized, returning 0 - thread=%p", utils_thread_id());
        log_info("Transcription already initialized");
        utils_mutex_unlock(ctx_mutex);
        return 0;
    }

    TranscriptionProvider preferred = transcription_get_provider();
    const TranscriptionProviderOps *ops = (preferred == TRANSCRIPTION_PROVIDER_REMOTE_HTTP)
        ? &kRemoteProvider
        : &kWhisperProvider;

    if (!ops->available) {
        log_info("Preferred provider '%s' unavailable, falling back to whisper", ops->name);
        ops = &kWhisperProvider;
    }

    g_provider_ops = ops;

    int result = ops->init(model_path);
    if (result != 0) {
        log_error("ERROR: Failed to initialize provider '%s'", ops->name);
        g_provider_ops = NULL;
        utils_mutex_unlock(ctx_mutex);
        return -1;
    }

    log_info("✅ Transcription provider ready: %s", ops->name);

    utils_mutex_unlock(ctx_mutex);
    return 0;
}

char *transcription_process(const float *audio_data, int n_samples, int sample_rate) {
    ensure_mutex_initialized();

    log_debug("transcription_process() ENTRY - thread=%p", utils_thread_id());

    if (audio_data == NULL || n_samples <= 0) {
        log_error("ERROR: Invalid parameters for transcription");
        return NULL;
    }

    utils_mutex_lock(ctx_mutex);
    log_debug("Acquired transcription mutex for processing - thread=%p", utils_thread_id());

    if (g_provider_ops == NULL) {
        log_debug("Provider not available - thread=%p", utils_thread_id());
        log_error("ERROR: Transcription not initialized");
        utils_mutex_unlock(ctx_mutex);
        return NULL;
    }

    log_info("🧠 Transcribing %d audio samples (%.2f seconds) via %s using language: %s\n",
             n_samples, (float) n_samples / 16000.0f, g_provider_ops->name, g_language);

    double total_start = utils_now();

    char *result = g_provider_ops->process(audio_data, n_samples, sample_rate);
    if (!result) {
        utils_mutex_unlock(ctx_mutex);
        return NULL;
    }

    result = normalize_transcription_text(result);

    double total_duration = utils_now() - total_start;

    if (result && strlen(result) > 0) {
        log_info("✅ Transcription complete: \"%s\"\n", result);
    } else {
        log_info("⚠️  No speech detected\n");
    }

    log_info("⏱️  Total transcription process took: %.0f ms\n", total_duration * 1000.0);

    log_debug("Releasing transcription mutex (normal completion) - thread=%p", utils_thread_id());
    utils_mutex_unlock(ctx_mutex);
    return result;
}

int transcribe_file(const char *audio_file, char *result, size_t result_size) {
    if (g_provider_ops == NULL) {
        log_error("ERROR: Transcription not initialized\n");
        return -1;
    }

    log_info("🎵 Loading audio file: %s\n", audio_file);

    // Proper WAV parser
    std::vector<float> pcmf32;

    std::ifstream file(audio_file, std::ios::binary);
    if (!file.is_open()) {
        log_error("ERROR: Could not open audio file: %s\n", audio_file);
        return -1;
    }

    // Read RIFF header
    char riff[4];
    uint32_t file_size;
    char wave[4];
    file.read(riff, 4);
    file.read(reinterpret_cast<char *>(&file_size), 4);
    file.read(wave, 4);

    if (strncmp(riff, "RIFF", 4) != 0 || strncmp(wave, "WAVE", 4) != 0) {
        log_error("ERROR: Not a valid WAV file\n");
        file.close();
        return -1;
    }

    // Find fmt chunk
    uint16_t format_tag = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size = 0;
    bool fmt_found = false;
    bool data_found = false;

    while (!fmt_found || !data_found) {
        char chunk_id[4];
        uint32_t chunk_size;

        if (!file.read(chunk_id, 4) || !file.read(reinterpret_cast<char *>(&chunk_size), 4)) {
            log_error("ERROR: Unexpected end of WAV file\n");
            file.close();
            return -1;
        }

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char *>(&format_tag), 2);
            file.read(reinterpret_cast<char *>(&channels), 2);
            file.read(reinterpret_cast<char *>(&sample_rate), 4);
            file.seekg(6, std::ios::cur); // skip byte_rate and block_align
            file.read(reinterpret_cast<char *>(&bits_per_sample), 2);
            file.seekg(chunk_size - 16, std::ios::cur); // skip any extra fmt data
            fmt_found = true;
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            data_found = true;

            // If data_size is 0, calculate actual size from file
            if (data_size == 0) {
                std::streampos current_pos = file.tellg();
                file.seekg(0, std::ios::end);
                std::streampos end_pos = file.tellg();
                data_size = (uint32_t) (end_pos - current_pos);
                file.seekg(current_pos); // return to data start
                log_info("🎵 Data chunk size was 0, calculated actual size: %u bytes\n", data_size);
            }

            break; // data chunk found, ready to read audio
        } else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!fmt_found || !data_found) {
        log_error("ERROR: Missing fmt or data chunk in WAV file\n");
        file.close();
        return -1;
    }

    log_info("🎵 WAV: %s, %dHz, %d channels, %d bits, format %d\n",
             audio_file, sample_rate, channels, bits_per_sample, format_tag);

    // Read audio data based on format
    uint32_t num_samples = data_size / (bits_per_sample / 8) / channels;
    pcmf32.reserve(num_samples);

    if (format_tag == 3 && bits_per_sample == 32) {
        // 32-bit float PCM
        for (uint32_t i = 0; i < num_samples; i++) {
            float sample_sum = 0.0f;

            // Read all channels and average them for mono output
            for (uint16_t ch = 0; ch < channels; ch++) {
                float sample;
                if (!file.read(reinterpret_cast<char *>(&sample), 4)) {
                    break;
                }
                sample_sum += sample;
            }

            if (file.fail()) break;
            pcmf32.push_back(sample_sum / channels); // Average channels for mono
        }
    } else if (format_tag == 1 && bits_per_sample == 16) {
        // 16-bit integer PCM
        for (uint32_t i = 0; i < num_samples; i++) {
            float sample_sum = 0.0f;

            // Read all channels and average them for mono output
            for (uint16_t ch = 0; ch < channels; ch++) {
                int16_t sample;
                if (!file.read(reinterpret_cast<char *>(&sample), 2)) {
                    break;
                }
                sample_sum += static_cast<float>(sample) / 32768.0f;
            }

            if (file.fail()) break;
            pcmf32.push_back(sample_sum / channels); // Average channels for mono
        }
    } else {
        log_error("ERROR: Unsupported WAV format (format=%d, bits=%d)\n", format_tag, bits_per_sample);
        file.close();
        return -1;
    }

    file.close();

    if (pcmf32.empty()) {
        log_error("ERROR: No audio data found in file: %s\n", audio_file);
        return -1;
    }

    log_info("🎵 Loaded %zu audio samples\n", pcmf32.size());

    // Convert to C-style array access for transcription
    const float *audio_data = pcmf32.data();
    int n_samples = (int) pcmf32.size();

    // Get dynamic transcription
    char *transcription = transcription_process(audio_data, n_samples, sample_rate);
    if (transcription == NULL) {
        return -1;
    }

    // Copy to result buffer for backward compatibility with test code
    strncpy(result, transcription, result_size - 1);
    result[result_size - 1] = '\0';

    // Warn if truncated
    if (strlen(transcription) >= result_size) {
        log_error("WARNING: Transcription truncated in transcribe_file - buffer too small\n");
    }

    free(transcription);
    return 0;
}

void transcription_cleanup(void) {
    ensure_mutex_initialized();

    utils_mutex_lock(ctx_mutex);

    if (g_provider_ops && g_provider_ops->cleanup) {
        g_provider_ops->cleanup();
    } else {
        whisper_provider_cleanup();
        remote_provider_cleanup();
    }

    g_provider_ops = NULL;

    utils_mutex_unlock(ctx_mutex);
}
