import Foundation
import MLX
import MLXNN
import MLXRandom

// attention

class F5Attention: Module {
    let dim: Int
    let heads: Int
    let innerDim: Int
    let dropout: Float

    let to_q: Linear
    let to_k: Linear
    let to_v: Linear
    let to_out: Sequential

    init(dim: Int, heads: Int = 8, dimHead: Int = 64, dropout: Float = 0.0) {
        self.dim = dim
        self.heads = heads
        self.innerDim = heads * dimHead
        self.dropout = dropout

        self.to_q = Linear(dim, innerDim)
        self.to_k = Linear(dim, innerDim)
        self.to_v = Linear(dim, innerDim)

        self.to_out = Sequential(layers: [
            Linear(innerDim, dim),
            Dropout(p: dropout)
        ])
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, rope: (MLXArray, Float)? = nil) -> MLXArray {
        let batch = x.shape[0]
        let seqLen = x.shape[1]

        var query = to_q(x)
        var key = to_k(x)
        var value = to_v(x)

        if let rope {
            let (freqs, xposScale) = rope
            let qXposScale = xposScale
            let kXposScale = pow(xposScale, -1.0)

            query = applyRotaryPosEmb(t: query, freqs: freqs, scale: qXposScale)
            key = applyRotaryPosEmb(t: key, freqs: freqs, scale: kXposScale)
        }

        query = rearrangeQuery(query, heads: heads)
        key = rearrangeQuery(key, heads: heads)
        value = rearrangeQuery(value, heads: heads)

        var attnMask: MLXArray? = nil
        if let mask = mask {
            let reshapedMask = mask.reshaped([mask.shape[0], 1, 1, mask.shape[1]])
            attnMask = MLX.repeated(reshapedMask, count: heads, axis: 1)
        }

        let scaleFactor = 1.0 / sqrt(Double(query.shape[query.shape.count - 1]))
        var output = MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value, scale: Float(scaleFactor), mask: attnMask)

        output = output.transposed(axes: [0, 2, 1, 3]).reshaped([batch, seqLen, -1])
        output = to_out(output)

        if let mask = mask {
            let maskReshaped = mask.reshaped([batch, seqLen, 1])
            output = MLX.where(maskReshaped, MLX.logicalNot(maskReshaped), 0.0)
        }

        return output
    }

    private func rearrangeQuery(_ query: MLXArray, heads: Int) -> MLXArray {
        let batchSize = query.shape[0]
        let seqLength = query.shape[1]
        let headDim = query.shape[2] / heads
        return query.reshaped([batchSize, seqLength, heads, headDim]).transposed(axes: [0, 2, 1, 3])
    }
}
