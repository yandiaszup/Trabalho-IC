////
////  WineClassification.swift
////  WineClassification
////
////  Created by Yan Dias on 26/06/19.
////  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
////
//
import Foundation


class WineClassificator {
    
    let dataParser = DataParser()
    
    var network: Network = Network(layerStructure: [11,22,10], learningRate: 0.1, momentum: 0.99, hasBias: true)
    
    var networkWithouMomentum: Network = Network(layerStructure: [11,22,10], learningRate: 0.1, momentum: 0.0, hasBias: true)
    
    var wineParameters: [[Double]] = [[Double]]()
    var wineClassifications: [[Double]] = [[Double]]()

    
    var outputs = [Double]()
    
    func start(precision: Double) {
        self.setInitialWeightsForSecondNetwork()
        
        guard let wineData = dataParser.parseData() else {
            return
        }
        
        splitWineData(data: wineData)
        wineClassifications = createTargetOutputVectors(outputs: outputs)
        
        print("backpropagation with momentum training results")
        trainNeuralNetwork(network: network, precision: precision)
        print("\n\nbackpropagation without momentum training results\n\n")
        trainNeuralNetwork(network: networkWithouMomentum, precision: precision)

        let result1 = self.network.validate(inputs: wineParameters, expecteds: wineClassifications)

        let result2 = self.networkWithouMomentum.validate(inputs: wineParameters, expecteds: wineClassifications)

        print("\n\ntotal: \(result1.total)\ncorrect: \(result1.correct)\n percentage: \(result1.percentage)\n\n")
        print("total: \(result2.total)\ncorrect: \(result2.correct)\n percentage: \(result2.percentage)")
        
        
    }
    
    func trainNeuralNetwork(network: Network, precision: Double) {
        var error = Double.infinity
        var numberOfCicles = 0
        
        var lasterror = 0.0
        var diference = 1.0
        
        let start = DispatchTime.now()
        while (diference > precision) {
            error = network.train(inputs: wineParameters.dropLast(10), expecteds: wineClassifications.dropLast(10))
            diference = abs(error - lasterror)
            lasterror = error
            numberOfCicles += 1
//            print(error)
        }
        let end = DispatchTime.now()
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
        
        print("Number of cicles \(numberOfCicles)")
        print("Number of epochs \(numberOfCicles * wineParameters.count)")
        print("Final EQM \(error)")
        print("Training time: \(Double(nanoTime)/1000000000) seconds")
    }
    
    func setInitialWeightsForSecondNetwork() {
        
        for i in 0...networkWithouMomentum.layers[1].neurons.count-1 {
            networkWithouMomentum.layers[1].neurons[i].weights = network.layers[1].neurons[i].weights
        }
        
        for i in 0...networkWithouMomentum.layers[2].neurons.count-1 {
            networkWithouMomentum.layers[2].neurons[i].weights = network.layers[2].neurons[i].weights
        }
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
