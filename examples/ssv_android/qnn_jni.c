#include <jni.h>
#include <stdint.h>
#include <stdlib.h>

_Static_assert(sizeof(int64_t) == 8, "int64 storage required");
_Static_assert(sizeof(__int128) == 16, "128-bit arithmetic required");
extern int qnn_input_dim(void);
extern int qnn_output_dim(void);
extern int qnn_encoder_channels(void);
extern int qnn_encoder_size(void);
extern int qnn_encode_bytes(const uint8_t *, int, int, int64_t *);
extern void qnn_forward_fixed(const int64_t *, int64_t *);

JNIEXPORT jlongArray JNICALL
Java_org_preqbmc_ssv_Qnn_forward(JNIEnv *env, jclass cls, jbyteArray rgb, jint h, jint w) {
    (void)cls;
    if (!rgb || h <= 0 || w <= 0 || h > 4096 || w > 4096 ||
        qnn_encoder_size() != qnn_input_dim() ||
        (int64_t)(*env)->GetArrayLength(env, rgb) != (int64_t)h * w * qnn_encoder_channels()) {
        jclass error = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
        if (error) (*env)->ThrowNew(env, error, "Expected contiguous HWC crop bytes");
        return NULL;
    }
    int ni = qnn_input_dim(), no = qnn_output_dim();
    int64_t *input = calloc((size_t)ni, sizeof(*input));
    int64_t *output = calloc((size_t)no, sizeof(*output));
    if (!input || !output) {
        free(input); free(output);
        jclass error = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
        if (error) (*env)->ThrowNew(env, error, "QNN buffers");
        return NULL;
    }
    jbyte *bytes = (*env)->GetByteArrayElements(env, rgb, NULL);
    if (!bytes) { free(input); free(output); return NULL; }
    int status = qnn_encode_bytes((const uint8_t *)bytes, h, w, input);
    (*env)->ReleaseByteArrayElements(env, rgb, bytes, JNI_ABORT);
    jlongArray result = NULL;
    if (status == 0) {
        qnn_forward_fixed(input, output);
        result = (*env)->NewLongArray(env, no);
        if (result) {
            for (int i = 0; i < no; ++i) {
                jlong value = (jlong)output[i];
                (*env)->SetLongArrayRegion(env, result, i, 1, &value);
            }
        }
    }
    free(input); free(output);
    return result;
}
