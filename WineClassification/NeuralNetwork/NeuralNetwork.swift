////
////  NeuralNetwork.swift
////  WineClassification
////
////  Created by Yan Dias on 24/06/19.
////  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
////
//
//import Foundation
//
//
//public extension ClosedRange where Bound: FloatingPoint {
//    public func random() -> Bound {
//        let range = self.upperBound - self.lowerBound
//        let randomValue = (Bound(arc4random_uniform(UINT32_MAX)) / Bound(UINT32_MAX)) * range + self.lowerBound
//        return randomValue
//    }
//}
//
//public class Layer {
//    
//    private var output: [Float]
//    private var input: [Float]
//    private var weights: [Float]
//    private var previousWeights: [Float]
//
//    init(inputSize: Int, outputSize: Int) {
//        self.output = [Float](repeating: 0, count: outputSize)
//        self.input = [Float](repeating: 0, count: inputSize + 1)
//        self.weights = (0..<(1 + inputSize) * outputSize).map { _ in
//            return (-2.0...2.0).random()
//        }
//        previousWeights = [Float](repeating: 0, count: weights.count)
//    }
//
//    public func run(inputArray: [Float]) -> [Float] {
//
//        for i in 0..<inputArray.count {
//            input[i] = inputArray[i]
//        }
//
//        input[input.count-1] = 1
//        var offSet = 0
//
//        for i in 0..<output.count {
//            for j in 0..<input.count {
//                output[i] += weights[offSet+j] * input[j]
//            }
//
//            output[i] = ActivationFunction.sigmoid(x: output[i])
//            offSet += input.count
//
//        }
//
//        return output
//    }
//
//    public func train(error: [Float], learningRate: Float, momentum: Float) -> [Float] {
//
//        var offset = 0
//        var nextError = [Float](repeating: 0, count: input.count)
//
//        for i in 0..<output.count {
//
//            let delta = error[i] * ActivationFunction.sigmoidDerivative(x: output[i])
//
//            for j in 0..<input.count {
//                let weightIndex = offset + j
//                nextError[j] = nextError[j] + weights[weightIndex] * delta
//                let dw = input[j] * delta * learningRate
//                weights[weightIndex] += previousWeights[weightIndex] * momentum + dw
//                previousWeights[weightIndex] = dw
//            }
//
//            offset += input.count
//        }
//
//        return nextError
//    }
//
//}
//
//public class ActivationFunction {
//
//    static func sigmoid(x: Float) -> Float {
//        return 1 / (1 + exp(-x))
//    }
//
//    static func sigmoidDerivative(x: Float) -> Float {
//        return x * (1 - x)
//    }
//
//}
//
//public class NeuralNetwork {
//
//    public static var learningRate: Float = 0.3
//    public static var momentum: Float = 0.6
//    public static var iterations: Int = 500
//
//    private var layers: [Layer] = []
//
//    public init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
//        self.layers.append(Layer(inputSize: inputSize, outputSize: hiddenSize))
//        self.layers.append(Layer(inputSize: hiddenSize, outputSize: outputSize))
//    }
//
//    public func run(input: [Float]) -> [Float] {
//
//        var activations = input
//
//        for i in 0..<layers.count {
//            activations = layers[i].run(inputArray: activations)
//        }
//
//        return activations
//    }
//
//    public func train(input: [Float], targetOutput: [Float], learningRate: Float, momentum: Float) {
//
//        let calculatedOutput = run(input: input)
//
//        var error = zip(targetOutput, calculatedOutput).map { $0 - $1 }
//
//        for i in (0...layers.count-1).reversed() {
//            error = layers[i].train(error: error, learningRate: learningRate, momentum: momentum)
//        }
//
//    }
//
//}
