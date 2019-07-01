//
//  WineClassification.swift
//  WineClassification
//
//  Created by Yan Dias on 26/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//

import Foundation

class WineClassification {
    
    let dataParser = DataParser()
    var trainingInputs = [[Float]]()
    var targetOutput = [[Float]]()
    var outputs = [Float]()
    
    func trainNeuralNetwork() {
        
        guard let wineData = dataParser.parseData() else {
            return
        }
        
        splitWineData(data: wineData)
        targetOutput = createTargetOutputVectors(outputs: outputs)
        
        let neuralNetwork = NeuralNetwork(inputSize: trainingInputs[0].count, hiddenSize: 22, outputSize: targetOutput[0].count)
        
        for _ in 0..<NeuralNetwork.iterations {
            for i in 0...targetOutput.count-1 {
                neuralNetwork.train(input: trainingInputs[i], targetOutput: targetOutput[i], learningRate: NeuralNetwork.learningRate, momentum: NeuralNetwork.momentum)
            }
        }
        
        print("\(targetOutput[1])")
        print("\(neuralNetwork.run(input: trainingInputs[1]))")
        print("\(targetOutput[3])")
        print("\(neuralNetwork.run(input: trainingInputs[3]))")
        print("\(targetOutput[7])")
        print("\(neuralNetwork.run(input: trainingInputs[7]))")
    }
    
    fileprivate func splitWineData(data: [[Float]]) {
        for x in data {
            var atributes = x
            let output = atributes.remove(at: 11)
            trainingInputs.append(atributes)
            outputs.append(output)
        }
    }
    
    fileprivate func createTargetOutputVectors(outputs: [Float]) -> [[Float]] {
        var targetOutputVectors = [[Float]]()
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
