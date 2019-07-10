//
//  ViewController.swift
//  WineClassification
//
//  Created by Yan Dias on 24/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//

import UIKit
import Charts

class ViewController: UIViewController {
    
    @IBOutlet weak var chart: LineChartView!
    @IBOutlet weak var trainingResultsLabel: UILabel!
    @IBOutlet weak var trainingResultsNoMomentumLabel: UILabel!
    @IBOutlet weak var learningRateLabel: UILabel!
    @IBOutlet weak var momentumLabel: UILabel!
    
    var trainingResults: (TrainingResults, TrainingResults)!
    
    let wineClassificator = WineClassificator()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        trainingResults = wineClassificator.start(precision: 0.000001) as! (TrainingResults, TrainingResults)

        createChart()
        updateTrainingResultsLabel()
        updateLearningRateAndMomentumLabels()
    }

    func updateLearningRateAndMomentumLabels() {
        self.learningRateLabel.text = "\(wineClassificator.network.learningRate)"
        self.momentumLabel.text = "\(wineClassificator.network.momentum)"
    }
    
    func updateTrainingResultsLabel() {
        trainingResultsLabel.text = "BackPropagation With Momentum Results:\n\nNumber of Cicles: \(trainingResults.0.numberOfCicles ?? 0)\nNumber of Epochs: \(trainingResults.0.numberOfEpochs ?? 0)\nEQM: \(trainingResults.0.finalEQM ?? 0.0)\nTraining Time: \(trainingResults.0.trainingTime ?? 0.0)"
        
        trainingResultsNoMomentumLabel.text = "BackPropagation Without Momentum Results:\n\nNumber of Cicles: \(trainingResults.1.numberOfCicles ?? 0)\nNumber of Epochs: \(trainingResults.1.numberOfEpochs ?? 0)\nEQM: \(trainingResults.1.finalEQM ?? 0.0)\nTraining Time: \(trainingResults.1.trainingTime ?? 0.0)"
        
    }
    
    func createChart() {
        
        var lineChartEntry = [ChartDataEntry]()
        var lineChartEntry2 = [ChartDataEntry]()
        
        let errorListMomentum = trainingResults.0.errorList
        let errorListWithoutMomentum = trainingResults.1.errorList
        
        chart.drawGridBackgroundEnabled = false
        chart.drawBordersEnabled = false
        chart.dragEnabled = false
        chart.isUserInteractionEnabled = false
        
        
        

        for i in 0...errorListMomentum!.count - 1 {
            let value = ChartDataEntry(x: Double(i), y: errorListMomentum![i])
            lineChartEntry.append(value)
        }
        
        for i in 0...errorListWithoutMomentum!.count - 1 {
            let value = ChartDataEntry(x: Double(i), y: errorListWithoutMomentum![i])
            lineChartEntry2.append(value)
        }
        
        let line1 = LineChartDataSet(entries: lineChartEntry, label: "Erros com momento")
        line1.drawCirclesEnabled = false
        line1.mode = .linear
        line1.colors = [NSUIColor.blue]
        
        let line2 = LineChartDataSet(entries: lineChartEntry2, label: "Erros sem momento")
        line2.drawCirclesEnabled = false
        line2.mode = .linear
        line2.colors = [NSUIColor.red]
        
        let data = LineChartData()
        data.addDataSet(line1)
        data.addDataSet(line2)
        chart.data = data
        
        chart.chartDescription?.text = "My awesome chart"

    }
}
