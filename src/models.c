#include "models.h"
#include "transcription.h"
#include "preferences.h"
#include "utils.h"
#include "overlay.h"
#include "logging.h"
#include "dialog.h"
#include "app.h"
#include <stdio.h>
#include <string.h>

// Helper function to extract filename from path
static const char *get_filename_from_path(const char *path) {
    if (!path) return "unknown";

    const char *filename = strrchr(path, '/');
    if (!filename) filename = strrchr(path, '\\');
    if (filename) filename++;
    else filename = path;
    return filename;
}

// THE ONE AND ONLY MODEL LOADING FUNCTION
int models_load(void) {
    log_info("Starting model loading at %.3f seconds", utils_now());

    // Cleanup existing model first
    transcription_cleanup();

    bool remote_provider = transcription_get_provider() == TRANSCRIPTION_PROVIDER_REMOTE_HTTP;
    overlay_show(remote_provider ? "Connecting to transcription service" : "Loading model");

    // Get the model path from preferences/bundled
    const char *model_path = utils_get_model_path();
    if (!model_path && !remote_provider) {
        overlay_hide();
        dialog_error("Model Error", "Could not find model file");
        return -1;
    }

    int result = transcription_init(model_path);

    if (result != 0) {
        if (remote_provider) {
            overlay_show_error("Failed to initialize transcription service");
            app_sleep_responsive(3000);
            overlay_hide();
            return -1;
        } else {
            // First failure - try fallback to base model
            const char *failed_model = preferences_get_string("model");
            char fallback_msg[256];

            if (failed_model && strlen(failed_model) > 0) {
                const char *filename = get_filename_from_path(failed_model);
                snprintf(fallback_msg, sizeof(fallback_msg), "Failed to load %s, falling back to base model", filename);

                // Remove corrupted user model
                log_info("Removing corrupted user model: %s", failed_model);
                remove(failed_model);
            } else {
                snprintf(fallback_msg, sizeof(fallback_msg), "Failed to load model, falling back to base model");
            }

            overlay_show_error(fallback_msg);

            // Clear the model from preferences to use bundled model
            preferences_set_string("model", "");
            preferences_save();

            // Wait, then try again with base model
            app_sleep_responsive(3000);

            overlay_show("Loading base model");
            model_path = utils_get_model_path(); // Get bundled model path
            if (model_path) {
                result = transcription_init(model_path);
            }

            if (!model_path || result != 0) {
                // Final failure
                char error_msg[256];
                if (model_path) {
                    const char *filename = get_filename_from_path(model_path);
                    snprintf(error_msg, sizeof(error_msg), "Failed to load %s", filename);
                } else {
                    snprintf(error_msg, sizeof(error_msg), "Model not found");
                }

                overlay_show_error(error_msg);
                app_sleep_responsive(3000);
                overlay_hide();
                return -1;
            }
        }
    }

    // Set language from preferences
    const char *language = preferences_get_string("language");
    transcription_set_language(language ? language : "en");

    // Model loaded successfully
    log_info("Model loaded successfully at %.3f seconds", utils_now());
    
    // Keep overlay visible for 1 second for user feedback (but not on startup)
    static bool is_startup = true;
    if (!is_startup) {
        app_sleep_responsive(1000);
    }
    is_startup = false;
    
    overlay_hide();
    return 0;
}

// Get VAD model path
const char *models_get_vad_path(void) {
    return utils_get_vad_model_path();
}

// Check if a model file exists
bool models_file_exists(const char *path) {
    if (!path) return false;
    
    FILE *file = fopen(path, "r");
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}
