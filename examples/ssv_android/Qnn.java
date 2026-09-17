package org.preqbmc.ssv;

/** Native generated C only: no TFLite, NNAPI, GPU or implicit bitmap resize. */
public final class Qnn {
    static { System.loadLibrary("preqbmc_ssv"); }
    private Qnn() {}
    public static native long[] forward(byte[] rgbHwc, int height, int width);
    public static int selectedClass(long[] logits) {
        if (logits == null || logits.length == 0) throw new IllegalArgumentException("Missing logits");
        int selected = 0;
        for (int j = 1; j < logits.length; ++j)
            if (logits[j] > logits[selected]) selected = j;
        return selected;
    }
}
