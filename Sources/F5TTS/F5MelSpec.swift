import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXFFT

public class F5MelSpec: Module {
    let sampleRate: Int
    let nFFT: Int
    let hopLength: Int
    let nMels: Int
    let filterbank: MLXArray

    init(
        sampleRate: Int = 24000,
        nFFT: Int = 1024,
        hopLength: Int = 256,
        nMels: Int = 100,
        filterbank: MLXArray
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels
        self.filterbank = filterbank
    }

    public func callAsFunction(x: MLXArray) -> MLXArray {
        logMelSpectrogram(audio: x, nMels: nMels, nFFT: nFFT, hopLength: hopLength, filterbank: filterbank)
    }

    public func stft(x: MLXArray, window: MLXArray, nperseg: Int, noverlap: Int? = nil, nfft: Int? = nil) -> MLXArray {
        let nfft = nfft ?? nperseg
        let noverlap = noverlap ?? nfft
        let padding = nperseg / 2
        let x = MLX.padded(x, width: IntOrPair(padding))
        let strides = [noverlap, 1]
        let t = (x.shape[0] - nperseg + noverlap) / noverlap
        let shape = [t, nfft]
        let stridedX = MLX.asStrided(x, shape, strides: strides)
        return MLXFFT.rfft(stridedX * window)
    }

    public func logMelSpectrogram(audio: MLXArray, nMels: Int = 100, nFFT: Int = 1024, hopLength: Int = 256, filterbank: MLXArray) -> MLXArray {
        let freqs = stft(x: audio, window: hanning(nFFT), nperseg: nFFT, noverlap: hopLength)
        let magnitudes = freqs[0..<freqs.shape[0] - 1].abs()
        let melSpec = MLX.matmul(magnitudes, filterbank.T)
        let logSpec = MLX.maximum(melSpec, 1e-5).log()
        return MLX.expandedDimensions(logSpec, axis: 0)
    }

    public func hanning(_ size: Int) -> MLXArray {
        let window = (0..<size).map { 0.5 * (1.0 - cos(2.0 * .pi * Double($0) / Double(size - 1))) }
        return MLXArray(converting: window)
    }
}
