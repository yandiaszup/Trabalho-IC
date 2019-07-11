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
    
    // precisao adotada
    let precision = 0.000001
    
    var network: Network = Network(layerStructure: [11,22,10], learningRate: 0.01, momentum: 0.7, hasBias: true)
    
    var networkWithouMomentum: Network = Network(layerStructure: [11,22,10], learningRate: 0.01, momentum: 0.0, hasBias: true)
    
    // Entradas
    var wineParameters: [[Double]] = [[Double]]()
    // Saidas esperadas
    var wineClassifications: [[Double]] = [[Double]]()

    var outputs = [Double]()
    
    func start() -> (TrainingResults?,TrainingResults?){
        
        fetchWineData()
        
        self.setInitialWeightsForSecondNetwork()
        
        print("backpropagation with momentum training results")
        let trainingWithMomentumResults = trainNeuralNetwork(network: network)
        print("\n\nbackpropagation without momentum training results\n\n")
        let trainingWithoutMomentumResults = trainNeuralNetwork(network: networkWithouMomentum)

        let result1 = self.network.validate(inputs: wineParameters, expecteds: wineClassifications)

        let result2 = self.networkWithouMomentum.validate(inputs: wineParameters, expecteds: wineClassifications)
        
        trainingWithMomentumResults.successRate = result1.percentage
        trainingWithoutMomentumResults.successRate = result2.percentage


        print("\nbackpropagation with momentum validation:\ntotal: \(result1.total)\ncorrect: \(result1.correct)\npercentage: \(result1.percentage)\n\nbackpropagation without momentum validation:\n")
        print("total: \(result2.total)\ncorrect: \(result2.correct)\npercentage: \(result2.percentage)")
        
        return (trainingWithMomentumResults, trainingWithoutMomentumResults)
    }
    
    // pega os valores do arquivo training e os aloca em WineClassifications e WineParameters
    func fetchWineData() {
        guard let wineData = dataParser.parseData() else {
            return
        }
        
        splitWineData(data: wineData)
        wineClassifications = createTargetOutputVectors(outputs: outputs)
    }
    
    // inicia o processo de treinamento para a rede dada
    func trainNeuralNetwork(network: Network) -> TrainingResults{
        var error = Double.infinity
        var numberOfCicles = 0
        
        var lasterror = 0.0
        var diference = 1.0
        
        var errorList = [Double]()
        
        let start = DispatchTime.now()
        while (diference > precision) {
            error = network.train(inputs: wineParameters.dropLast(15), expecteds: wineClassifications.dropLast(15))
            diference = abs(error - lasterror)
            lasterror = error
            numberOfCicles += 1
            errorList.append(error)
        }
        let end = DispatchTime.now()
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
        
        print("Number of cicles \(numberOfCicles)")
        print("Number of epochs \(numberOfCicles * wineParameters.dropLast(15).count)")
        print("Final EQM \(error)")
        print("Training time: \(Double(nanoTime)/1000000000) seconds")
        
        let trainingResults = TrainingResults(numberOfCicles: numberOfCicles, numberOfEpochs: numberOfCicles * wineParameters.dropLast(15).count, finalEQM: error, trainingTime: Double(nanoTime)/1000000000, errorList: errorList)
        
        return trainingResults
    }

    // torna os pesos iniciais da segunda rede iguais ao da primeira (para comparaca dos algoritmos com momento e sem momento)
    func setInitialWeightsForSecondNetwork() {
        for i in 0...networkWithouMomentum.layers[1].neurons.count-1 {
            networkWithouMomentum.layers[1].neurons[i].weights = network.layers[1].neurons[i].weights
        }

        for i in 0...networkWithouMomentum.layers[2].neurons.count-1 {
            networkWithouMomentum.layers[2].neurons[i].weights = network.layers[2].neurons[i].weights
        }
    }
    
    
    fileprivate func splitWineData(data: [[Double]]) {
        for x in data {
            var atributes = x
            let output = atributes.remove(at: 11)
            wineParameters.append(atributes)
            outputs.append(output)
        }
    }
    
    // cria vetores que serao usados como valores experados para a rede (transforma uma nota em um vetor)
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

class TrainingResults {
    let numberOfCicles: Int!
    let numberOfEpochs: Int!
    let finalEQM: Double!
    let trainingTime: Double!
    let errorList: [Double]!
    var successRate = 0.0
    
    init(numberOfCicles : Int, numberOfEpochs: Int, finalEQM: Double, trainingTime: Double, errorList: [Double]) {
        self.numberOfCicles = numberOfCicles
        self.numberOfEpochs = numberOfEpochs
        self.finalEQM = finalEQM
        self.trainingTime = trainingTime
        self.errorList = errorList
    }
}
