import Foundation
import MLX
import MLXNN
import MLXRandom

// ConvNeXt-v2 block

open class F5GroupableConv1d: Module, UnaryLayer {
    public let weight: MLXArray
    public let bias: MLXArray?
    public let padding: Int
    public let dilation: Int
    public let groups: Int
    public let stride: Int

    convenience init(_ inputChannels: Int, _ outputChannels: Int, kernelSize: Int, padding: Int, dilation: Int, groups: Int) {
        self.init(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: kernelSize, padding: padding, dilation: dilation, groups: groups)
    }

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))

        self.weight = uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels / groups]
        )
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}
