//
//  Network.swift
//  WineClassification
//
//  Created by Yan Dias on 24/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//

import Foundation // for sqrt

class Network {
    var layers: [Layer]
    
    init(layerStructure:[Int], activationFunction: @escaping (Double) -> Double = sigmoid, derivativeActivationFunction: @escaping (Double) -> Double = derivativeSigmoid, learningRate: Double = 0.25, momentum: Double, hasBias: Bool = false) {
        if (layerStructure.count < 3) {
            print("Error: Should be at least 3 layers (1 input, 1 hidden, 1 output)")
        }
        layers = [Layer]()
        // Camadas de entrada
        layers.append(Layer(numNeurons: layerStructure[0], activationFunction: activationFunction, derivativeActivationFunction: derivativeActivationFunction, learningRate: learningRate, momentum: momentum, hasBias: hasBias))
        
        // Camadas escondidas
        for x in layerStructure.enumerated() where x.offset != 0 && x.offset != layerStructure.count - 1 {
            layers.append(Layer(previousLayer: layers[x.offset - 1], numNeurons: x.element, activationFunction: activationFunction, derivativeActivationFunction: derivativeActivationFunction, learningRate: learningRate, momentum: momentum, hasBias: hasBias))
        }
        
        // Camada de saida
        layers.append(Layer(previousLayer: layers[layerStructure.count - 2], numNeurons: layerStructure.last!, activationFunction: activationFunction, derivativeActivationFunction: derivativeActivationFunction, learningRate: learningRate, momentum: momentum, hasBias: false))
    }
    
    /// Processa uma entrada dada
    func outputs(input: [Double]) -> [Double] {
        return layers.reduce(input) { $1.outputs(inputs: $0) }
    }
    
    /// Calcula os deltas de cada neuronio
    func backPropagate(expected: [Double]) {
        //calculate delta for output layer neurons
        layers.last?.calculateDeltasForOutputLayer(expected: expected)
        //calculate delta for prior layers
        for l in (1..<layers.count - 1).reversed() { // skip input layer
            layers[l].calculateDeltasForHiddenLayer(nextLayer: layers[l + 1])
        }
    }
    
    /// Corrige os pesos dos neuronios a partir do delta
    func updateWeights() {
        for layer in layers.dropFirst() { // skip input layer
            for neuron in layer.neurons {
                for w in 0..<neuron.weights.count {
                    neuron.weights[w] = neuron.weights[w] + ((layer.previousLayer?.outputCache[w])! * neuron.delta)
                }
            }
        }
    }
    
    /// comeca o processo de trainamento da rede
    func train(inputs:[[Double]], expecteds:[[Double]], printError:Bool = true) -> Double{
        var err: Double = 0.0
        for (location, xs) in inputs.enumerated() {
            let ys = expecteds[location]
            let outs = outputs(input: xs)
            if (printError) {
                    let diff = sub(x: outs, y: ys)
                    let error = sum(x: mul(x: diff, y: diff))
                    err += error
//                    print("\(error)")
//                }
            }
            backPropagate(expected: ys)
            updateWeights()
        }
        err = err / Double(inputs.count)
//        print(err)
        return err
    }
    
    
    func interpretOutput(output: [Double]) -> [Double] {
        var maxValue: Double = 0.0
        var maxValueIndex: Int = 0
        for index in 0...output.count-1 {
            let value = output[index]
            if value >= maxValue {
                maxValue = value
                maxValueIndex = index
            }
        }
        
        var outputs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        outputs[maxValueIndex] = 1.0
        return outputs
    }
    
    /// Valida um conjunto de amostras
    func validate(inputs:[[Double]], expecteds:[[Double]]) -> (correct: Int, total: Int, percentage: Double) {
        var correct = 0
        for (input, expected) in zip(inputs, expecteds) {
            let result = interpretOutput(output: outputs(input: input))
            if result == expected {
                correct += 1
            }
        }
        let percentage = Double(correct) / Double(inputs.count)
        return (correct, inputs.count, percentage)
    }
    
    func evaluateWine(input:[Double]) -> [Double] {
        let resultVector = interpretOutput(output: outputs(input: input))
        
        let score = interpretOutput(output: resultVector)
        
        return score
    }
    
    func printCurrentWeights() {
        print("\n\n\n----------------Hidden Layer Weights----------------\n\n\n")
        self.layers[1].neurons.forEach { (neuron) in
            print("N: ", terminator: "")
            neuron.weights.forEach({ (weight) in
                print(String(format: "%.4f, ", weight), terminator: "")
            })
            print("\n")
        }
        print("\n\n\n----------------Output Layer Weights----------------\n\n\n")
        self.layers[2].neurons.forEach { (neuron) in
            print("N: ", terminator: "")
            neuron.weights.forEach({ (weight) in
                print(String(format: "%.4f, ", weight), terminator: "")
            })
            print("\n")
        }
    }
}
