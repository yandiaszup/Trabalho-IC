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
    var momentum: Double = 0.0
    var learningRate: Double = 0.0
    
    init(layerStructure:[Int], activationFunctionForHiddenLayer: @escaping (Double) -> Double = relu, activationFunctionForOutputLayer: @escaping (Double) -> Double = sigmoid, derivativeActivationFunctionForHiddenLayer: @escaping (Double) -> Double = derivateRelu, derivativeActivationFunctionForOutputLayer: @escaping (Double) -> Double = derivativeSigmoid, learningRate: Double = 0.25, momentum: Double, hasBias: Bool = false) {
        if (layerStructure.count < 3) {
            print("Error: Should be at least 3 layers (1 input, 1 hidden, 1 output)")
        }
        layers = [Layer]()
        // Camadas de entrada
        layers.append(Layer(numNeurons: layerStructure[0], activationFunction: activationFunctionForHiddenLayer, derivativeActivationFunction: derivativeActivationFunctionForHiddenLayer, hasBias: hasBias))
        
        // Camadas escondidas
        for x in layerStructure.enumerated() where x.offset != 0 && x.offset != layerStructure.count - 1 {
            layers.append(Layer(previousLayer: layers[x.offset - 1], numNeurons: x.element, activationFunction: activationFunctionForHiddenLayer, derivativeActivationFunction: derivativeActivationFunctionForHiddenLayer, hasBias: hasBias))
        }
        
        // Camada de saida
        layers.append(Layer(previousLayer: layers[layerStructure.count - 2], numNeurons: layerStructure.last!, activationFunction: activationFunctionForOutputLayer, derivativeActivationFunction: derivativeActivationFunctionForOutputLayer, hasBias: false))
        
        self.learningRate = learningRate
        self.momentum = momentum
    }
    
    /// Processa uma entrada dada e retorna a saida do pmc
    func outputs(input: [Double]) -> [Double] {
        return layers.reduce(input) { $1.outputs(inputs: $0) }
    }
    
    /// Backpropagation inicia pela camada de saida ate a primeira camada escondida
    func backPropagate(expected: [Double]) {
        layers.last?.calculateDeltasForOutputLayer(expected: expected)
        
        for l in (1..<layers.count - 1).reversed() { // skip input layer
            layers[l].calculateDeltasForHiddenLayer(nextLayer: layers[l + 1])
        }
    }
    
    /// Corrige os pesos dos neuronios a partir do gradiente calculado em backpropagate
    func updateWeights() {
        for layer in layers.dropFirst() { // skip input layer
            for neuron in layer.neurons {
                for w in 0..<neuron.weights.count {
                    let delta = (self.learningRate * neuron.gradient * (layer.previousLayer?.outputCache[w])!) + (self.momentum * neuron.lastDelta[w])
                    neuron.lastDelta[w] = delta
                    neuron.weights[w] = neuron.weights[w] + delta
                }
            }
        }
    }
    
    // treina a rede para um conjunto de dados dado
    func train(inputs:[[Double]], expecteds:[[Double]]) -> Double{
        var err: Double = 0.0
        for (location, xs) in inputs.enumerated() {
            let ys = expecteds[location]
            let outs = outputs(input: xs)
            
            let diff = sub(x: outs, y: ys)
            let error = sum(x: mul(x: diff, y: diff))
            err += error
            //                    print("\(error)")
        
            backPropagate(expected: ys)
            updateWeights()
        }
        err = err / Double(inputs.count)
//        print(err)
        return err
    }
    
    // Defuzzy (transforma um vetor de probabilidades em um vetor de 0 e 1)
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
