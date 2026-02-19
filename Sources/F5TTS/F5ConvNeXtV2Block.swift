import Foundation
import MLX
import MLXNN
import MLXRandom

class F5ConvNeXtV2Block: Module, UnaryLayer {
    let dwconv: F5GroupableConv1d
    let norm: LayerNorm
    let pwconv1: Linear
    let act: GELU
    let grn: GRN
    let pwconv2: Linear

    init(dim: Int, intermediateDim: Int, dilation: Int = 1) {
        let padding = (dilation * (7 - 1)) / 2
        self.dwconv = F5GroupableConv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: padding, groups: dim)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6)
        self.pwconv1 = Linear(dim, intermediateDim)
        self.act = GELU()
        self.grn = GRN(dim: intermediateDim)
        self.pwconv2 = Linear(intermediateDim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = dwconv(x)
        out = norm(out)
        out = pwconv1(out)
        out = act(out)
        out = grn(out)
        out = pwconv2(out)
        return residual + out
    }
}
