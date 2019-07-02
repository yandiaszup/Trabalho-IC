////
////  WineClassification.swift
////  WineClassification
////
////  Created by Yan Dias on 26/06/19.
////  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
////
//
//import Foundation
//

class WineClassificator {
    
    let dataParser = DataParser()
    
    var network: Network = Network(layerStructure: [11,22,10], learningRate: 0.2, momentum: 0.7, hasBias: true)
    
    // for training
    var wineParameters: [[Double]] = [[Double]]()
    var wineClassifications: [[Double]] = [[Double]]()
    
    let numberOfIteractions = 700
    
    var outputs = [Double]()
    
    
    func trainNeuralNetwork() {
        
        guard let wineData = dataParser.parseData() else {
            return
        }
        
        splitWineData(data: wineData)
        wineClassifications = createTargetOutputVectors(outputs: outputs)
        
        for _ in 0...numberOfIteractions {
            network.train(inputs: wineParameters, expecteds: wineClassifications)
        }
        
        let result = network.validate(inputs: wineParameters, expecteds: wineClassifications)
        print(result.correct)
        print(result.percentage)
        print(result.total)
        
//        let testInput = [7.6,0.49,0.26,1.6,0.236,10,88,0.9968,3.11,0.8,9.3]
//
//        let result1 = network.evaluateWine(input: normalizeInputVector(input: testInput))
//        print(result1)
    }
    
    fileprivate func normalizeInputVector(input: [Double]) -> [Double] {
        var normalizedInput = [Double]()
        let minMaxList = dataParser.minMax
        for i in 0...10 {
            let value = input[i]
            let normalizedValue = dataParser.normalizedValue(value: value, maxValue: minMaxList[i].1, minValue: minMaxList[i].0)
            normalizedInput.append(normalizedValue)
        }
        return normalizedInput
    }
    
    
    fileprivate func splitWineData(data: [[Double]]) {
        for x in data {
            var atributes = x
            let output = atributes.remove(at: 11)
            wineParameters.append(atributes)
            outputs.append(output)
        }
    }
    
    fileprivate func createTargetOutputVectors(outputs: [Double]) -> [[Double]] {
        var targetOutputVectors = [[Double]]()
        for output in outputs {
            switch output {
            case 0.0/9.0:
                targetOutputVectors.append([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 1.0/9.0:
                targetOutputVectors.append([0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 2.0/9.0:
                targetOutputVectors.append([0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 3.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 4.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
                break
            case 5.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0])
                break
            case 6.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0])
                break
            case 7.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0])
                break
            case 8.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
                break
            case 9.0/9.0:
                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
                break
            default:
                break
            }
        }
        return targetOutputVectors
    }
    
    
}

//class WineClassification {
//
//    let dataParser = DataParser()
//    var trainingInputs = [[Float]]()
//    var targetOutput = [[Float]]()
//    var outputs = [Float]()
//
//    func trainNeuralNetwork() {
//
//        guard let wineData = dataParser.parseData() else {
//            return
//        }
//
//        splitWineData(data: wineData)
//        targetOutput = createTargetOutputVectors(outputs: outputs)
//
//        let neuralNetwork = NeuralNetwork(inputSize: trainingInputs[0].count, hiddenSize: 22, outputSize: targetOutput[0].count)
//
//        for _ in 0..<NeuralNetwork.iterations {
//            for i in 0...targetOutput.count-1 {
//                neuralNetwork.train(input: trainingInputs[i], targetOutput: targetOutput[i], learningRate: NeuralNetwork.learningRate, momentum: NeuralNetwork.momentum)
//            }
//        }
//
//        print("\(targetOutput[1])")
//        print("\(neuralNetwork.run(input: trainingInputs[1]))")
//        print("\(targetOutput[3])")
//        print("\(neuralNetwork.run(input: trainingInputs[3]))")
//        print("\(targetOutput[7])")
//        print("\(neuralNetwork.run(input: trainingInputs[7]))")
//    }
//
//    fileprivate func splitWineData(data: [[Float]]) {
//        for x in data {
//            var atributes = x
//            let output = atributes.remove(at: 11)
//            trainingInputs.append(atributes)
//            outputs.append(output)
//        }
//    }
//
//    fileprivate func createTargetOutputVectors(outputs: [Float]) -> [[Float]] {
//        var targetOutputVectors = [[Float]]()
//        for output in outputs {
//            switch output {
//            case 0.0/9.0:
//                targetOutputVectors.append([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
//                break
//            case 1.0/9.0:
//                targetOutputVectors.append([0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
//                break
//            case 2.0/9.0:
//                targetOutputVectors.append([0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
//                break
//            case 3.0/9.0:
//                targetOutputVectors.append([0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0])
//                break
//            case 4.0/9.0:
//                targetOutputVectors.append([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0])
//                break
//            case 5.0/9.0:
//                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0])
//                break
//            case 6.0/9.0:
//                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0])
//                break
//            case 7.0/9.0:
//                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0])
//                break
//            case 8.0/9.0:
//                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
//                break
//            case 9.0/9.0:
//                targetOutputVectors.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
//                break
//            default:
//                break
//            }
//        }
//        return targetOutputVectors
//    }
//
//    fileprivate func generateRate(withOutputVector outputVector: [Float]) {
//        var maxValue: Float = 0.0
//        var maxValueIndex: Int = 0
//        for index in 0...outputVector.count-1 {
//            let value = outputVector[index]
//            if value > maxValue {
//                maxValue = value
//                maxValueIndex = index
//            }
//        }
//    }
//
//}
