//
//  ViewController.swift
//  WineClassification
//
//  Created by Yan Dias on 24/06/19.
//  Copyright © 2019 Yan lucas damasceno dias. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        let wineClassificator = WineClassificator()
        wineClassificator.trainNeuralNetwork()
        
    }
}
