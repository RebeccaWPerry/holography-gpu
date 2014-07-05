#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
'''
Graphical user interface to explore how single sphere holograms
change with the various parameters that they depend on.

This can be run from a terminal as:
optirun python gui_manipulate.py

Can also be run from within an ipython session as:
run gui_manipulate

.. moduleauthor:: Rebecca W. Perry <perry.becca@gmail.com>
'''

import sys
from PyQt4 import QtGui, QtCore
from PIL.ImageQt import ImageQt

import holopy as hp
from holopy.scattering.scatterer import Sphere, Spheres
from holopy.core import ImageSchema, Optics
from holopy.scattering.theory import Mie, Multisphere, gpuMie
from holopy.propagation import propagate

import scipy
import numpy as np
import Image
import time


#TODO: how to deal with two z's for reconstructing in dimer hologram case?

class Holo(QtGui.QWidget):
    '''
    Display single or double sphere hologram with interactive capability
    to modify parameters used to calculate the hologram.
    '''
    
    def __init__(self):
        super(Holo, self).__init__()
        self.initUI()

    def initUI(self):

        #main image
        self.label = QtGui.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        #to display syntax of sphere and schema
        spheretitle = QtGui.QLabel()
        spheretitle.setText('Holopy Scatterer Syntax:')
        spheretitle.setStyleSheet('font-weight:bold')
        spheretitle.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)
        self.sphObject = QtGui.QLabel(self)
        self.sphObject.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)
        self.sphObject.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.schemaObject = QtGui.QLabel(self)
        self.schemaObject.setWordWrap(True)
        self.schemaObject.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.lastZ=0
        self.lastholo=2

        self.oldMieCPUholo = 0
        self.oldMieCPUparameters = 0 
        self.oldMieCPUschema = 0

        self.oldMieGPUHolo = 0
        self.oldscattererparameters = 0
        self.oldMieGPUschema = 0

        self.oldMultiHolo = 0
        self.oldMultiparameters = 0
        self.oldMultischema = 0

        self.oldReconParameters = 0
        self.oldReconHolo = 0

        #warning to be used when hologram calculation is not feasible
        self.warning = QtGui.QLabel(self)
        self.warning.setText('')
        self.warning.setStyleSheet('font-size: 20pt; color: red')
        self.warning.setGeometry(30,10,350,100)
        self.warning.setWordWrap(True)

        self.maxz_diameters = 398 #actually the range, two less than the maximum, in units of radii

        #schema adjustment controls
        schemacontrol = QtGui.QHBoxLayout()

        width = QtGui.QLabel(self)
        width.setText('Width (px):')
        self.widthText = QtGui.QLineEdit(self)
        self.widthText.setText('256')
        self.widthText.setFixedWidth(80)
        self.widthText.textChanged.connect(self.rangeChange2)
        self.widthText.textChanged.connect(self.calculateHologram)

        height = QtGui.QLabel(self)
        height.setText('Height (px):')
        self.heightText = QtGui.QLineEdit(self)
        self.heightText.setText('256')
        self.heightText.setFixedWidth(80)
        self.heightText.textChanged.connect(self.rangeChange2)
        self.heightText.textChanged.connect(self.calculateHologram)

        wave = QtGui.QLabel(self)
        wave.setText('Wavelength:')
        self.waveText = QtGui.QLineEdit(self)
        self.waveText.setText('.660')
        self.waveText.setFixedWidth(80)
        self.waveText.textChanged.connect(self.inmedChange)
        self.waveText.textChanged.connect(self.calculateHologram)

        mindex = QtGui.QLabel(self)
        mindex.setText('Medium  Index:')
        self.mindexText = QtGui.QLineEdit(self)
        self.mindexText.setText('1.33')
        self.mindexText.setFixedWidth(80)
        self.mindexText.textChanged.connect(self.inmedChange)
        self.mindexText.textChanged.connect(self.calculateHologram)

        pxsize = QtGui.QLabel(self)
        pxsize.setText('Pixel Spacing:')
        self.pxsizeText = QtGui.QLineEdit(self)
        self.pxsizeText.setText('0.1')
        self.pxsizeText.setFixedWidth(80)
        self.scale = float(self.pxsizeText.text()) #to be used setting up sliders and lcd's
        self.pxsizeText.textChanged.connect(self.rangeChange)
        self.pxsizeText.textChanged.connect(self.calculateHologram)

        #number of spheres control (how to extend to many spheres?)
        self.onesphere = QtGui.QCheckBox(self)
        self.onesphere.setText('1 sphere')
        self.twospheres = QtGui.QCheckBox(self)
        self.twospheres.setText('2 spheres')
        self.onesphere.setChecked(True)
        self.onesphere.stateChanged.connect(self.resetLayout)
        self.onesphere.stateChanged.connect(self.calculateHologram)
        numspheres = QtGui.QButtonGroup(self)
        numspheres.addButton(self.onesphere)
        numspheres.addButton(self.twospheres)


        #sphere parameter adjustment controls
        x = QtGui.QLabel(self)
        x.setText('x')
        x.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        self.lcd = QtGui.QLCDNumber(self)
        self.lcd.setGeometry(470,10,100,30)
        self.lcd.setMinimumSize(1,30)
        self.lcd.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.lcd.setStyleSheet('color: black')

        start = 10
        self.lcd.display(start)

        self.sld = QtGui.QSlider(QtCore.Qt.Horizontal,self)
        self.sld.setGeometry(470,40,100,30)
        self.sld.setMinimum(0)
        self.sld.setMaximum(256)
        self.sld.setSliderPosition(start/float(self.pxsizeText.text()))
        self.sld.valueChanged.connect(self.updateLCD)
        self.sld.sliderReleased.connect(self.calculateHologram)
        self.sld.valueChanged.connect(self.calculateHologram)
        self.sld.sliderPressed.connect(self.activate)


        y = QtGui.QLabel(self)
        y.setText('y')
        y.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        self.lcd2 = QtGui.QLCDNumber(self)
        self.lcd2.setGeometry(470, 80,100,30)
        self.lcd2.setMinimumSize(1,30)
        self.lcd2.display(start)
        self.lcd2.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.lcd2.setStyleSheet('color: black')


        self.sld2 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld2.setGeometry(470,110,100,30)
        self.sld2.setMinimum(0)
        self.sld2.setMaximum(256)
        self.sld2.setSliderPosition(start/float(self.pxsizeText.text()))
        self.sld2.valueChanged.connect(self.updateLCD)
        self.sld2.sliderReleased.connect(self.calculateHologram)
        self.sld2.valueChanged.connect(self.calculateHologram)
        self.sld2.sliderPressed.connect(self.activate)


        radius = QtGui.QLabel(self)
        radius.setText('radius')
        radius.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        start = .5
        self.lcd4 = QtGui.QLCDNumber(self)
        self.lcd4.setGeometry(470, 220,100,30)
        self.lcd4.display(start)
        self.lcd4.setMinimumSize(1,30)
        self.lcd4.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.lcd4.setStyleSheet('color: black')


        self.sld4 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld4.setGeometry(470,250,100,30)
        self.sld4.setMinimum(0)
        self.sld4.setMaximum(100)
        self.sld4.setMinimumSize(250,20)
        self.sld4.setSliderPosition((start-.25)*100)
        self.sld4.valueChanged.connect(self.updateLCD)
        self.sld4.valueChanged.connect(self.lengthscaleChange)
        self.sld4.valueChanged.connect(self.calculateHologram)
        self.sld4.sliderReleased.connect(self.calculateHologram)
        self.sld4.sliderPressed.connect(self.activate)


        z = QtGui.QLabel(self)
        z.setText('z')
        z.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        start = 30
        self.lcd3 = QtGui.QLCDNumber(self)
        self.lcd3.setGeometry(470,150,100,30)
        self.lcd3.display(15)
        self.lcd3.setMinimumSize(1,30)
        self.lcd3.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.lcd3.setStyleSheet('color: black')

        self.sld3 = QtGui.QSlider(QtCore.Qt.Horizontal, self)

        #z from twice the radius to 60 radii
        zlow = round(2*self.lcd4.value(),1)
        zhigh = zlow+round(self.maxz_diameters*self.lcd4.value(),1)

        self.sld3.setMinimum(0)
        self.sld3.setMaximum((zhigh-zlow)*10)
        self.sld3.setSliderPosition(round((start-zlow)/(zhigh-zlow)*self.sld3.maximum(),0))
        self.sld3.setGeometry(470,180,100,30)
        self.sld3.valueChanged.connect(self.updateLCD)
        self.sld3.valueChanged.connect(self.calculateHologram)
        self.sld3.sliderReleased.connect(self.calculateHologram)
        self.sld3.sliderPressed.connect(self.activate)


        #index goes from 1.00 to 3.00
        index = QtGui.QLabel(self)
        index.setText('index')
        index.setWordWrap(True)
        index.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        start = 1.6
        self.lcd5 = QtGui.QLCDNumber(self)
        self.lcd5.setGeometry(470,290,100,30)
        self.lcd5.display(1.60)
        self.lcd5.setMinimumSize(1,30)
        self.lcd5.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.sld5 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld5.setGeometry(470,320,100,30)
        self.sld5.setMinimum(0)
        self.sld5.setMinimumSize(250,20)
        self.sld5.setMaximum(200)
        self.sld5.setSliderPosition((start-1.0)/2.0*200)
        self.sld5.valueChanged.connect(self.updateLCD)
        self.sld5.valueChanged.connect(self.calculateHologram)
        self.sld5.sliderReleased.connect(self.calculateHologram)
        self.sld5.sliderPressed.connect(self.activate)


        #sphere parameter adjustment controls
        self.x2 = QtGui.QLabel(self)
        self.x2.setText('x2')
        self.x2.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        self.lcd7 = QtGui.QLCDNumber(self)
        self.lcd7.setGeometry(470,10,100,30)
        self.lcd7.setMinimumSize(1,30)
        self.lcd7.setSegmentStyle(QtGui.QLCDNumber.Flat)
        start = 120
        self.lcd7.display(start*self.scale)
        self.lcd7.setStyleSheet('color: black')


        self.sld7 = QtGui.QSlider(QtCore.Qt.Horizontal,self)
        self.sld7.setGeometry(470,40,100,30)
        self.sld7.setMinimum(0)
        self.sld7.setMaximum(256)
        self.sld7.setSliderPosition(start)
        self.sld7.valueChanged.connect(self.updateLCD)
        self.sld.sliderReleased.connect(self.calculateHologram)
        self.sld7.valueChanged.connect(self.calculateHologram)
        self.sld7.sliderPressed.connect(self.activate)


        self.y2 = QtGui.QLabel(self)
        self.y2.setText('y2')
        self.y2.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        self.lcd8 = QtGui.QLCDNumber(self)
        self.lcd8.setGeometry(470, 80,100,30)
        self.lcd8.setMinimumSize(1,30)
        self.lcd8.display(start*self.scale)
        self.lcd8.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.lcd8.setStyleSheet('color: black')


        self.sld8 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld8.setGeometry(470,110,100,30)
        self.sld8.setMinimum(0)
        self.sld8.setMaximum(256)
        self.sld8.setSliderPosition(start)
        self.sld8.valueChanged.connect(self.updateLCD)
        self.sld8.sliderReleased.connect(self.calculateHologram)
        self.sld8.valueChanged.connect(self.calculateHologram)
        self.sld8.sliderPressed.connect(self.activate)


        self.z2 = QtGui.QLabel(self)
        self.z2.setText('z2')
        self.z2.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignCenter)

        start = 35
        self.lcd9 = QtGui.QLCDNumber(self)
        self.lcd9.setGeometry(470,150,100,30)
        self.lcd9.display(35)
        self.lcd9.setMinimumSize(1,30)
        self.lcd9.setSegmentStyle(QtGui.QLCDNumber.Flat)
        self.lcd9.setStyleSheet('color: black')


        self.sld9 = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld9.setGeometry(470,180,100,30)
        self.sld9.setMinimum(0)
        self.sld9.setMaximum((zhigh-zlow)*10)
        self.sld9.setSliderPosition(round((start-zlow)/(zhigh-zlow)*self.sld9.maximum(),0))
        self.sld9.valueChanged.connect(self.updateLCD)
        self.sld9.valueChanged.connect(self.calculateHologram)
        self.sld9.valueChanged.connect(self.activate)
        self.sld9.sliderReleased.connect(self.calculateHologram)
        self.sld9.sliderPressed.connect(self.activate)


        #calculation timer
        self.timer = QtGui.QLabel(self)
        self.timer.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)


        #scattering theories
        self.gpu = QtGui.QPushButton('GPU Mie', self)
        self.gpu.setCheckable(True)
        self.gpu.setFixedHeight(40) #attribute from qwidget class
        self.gpu.clicked.connect(self.calculateHologram)

        self.cpu = QtGui.QPushButton('CPU Mie', self)
        self.cpu.setCheckable(True)
        self.cpu.setFixedHeight(40) #attribute from qwidget class
        self.cpu.clicked.connect(self.calculateHologram)

        self.multisphere = QtGui.QPushButton('CPU Multisphere', self)
        self.multisphere.setCheckable(True)
        self.multisphere.setFixedHeight(40) #attribute from qwidget class
        self.multisphere.clicked.connect(self.calculateHologram)

        self.reconstruct = QtGui.QPushButton('Reconstruct', self)
        self.reconstruct.setCheckable(True)
        self.reconstruct.setFixedHeight(40) #attribute from qwidget class
        self.reconstruct.clicked.connect(self.setuprecon)
        self.reconstruct.clicked.connect(self.calculateHologram)


        theories = QtGui.QButtonGroup(self)
        theories.addButton(self.gpu)
        theories.addButton(self.cpu)
        theories.addButton(self.multisphere)
        theories.addButton(self.reconstruct)

        #calculate starting image
        self.oldscatterer = Sphere(center=(1,1,1),n=1.4,r=.5)
        self.oldschema = Sphere(center=(1,1,1),n=1.4,r=.5)
        self.oltheory = 'none'
        self.calculateHologram()

        #################
        #GUI LAYOUT
        #################

        #number of spheres check box
        hbox0a = QtGui.QHBoxLayout()
        hbox0a.addWidget(self.onesphere)
        hbox0a.addWidget(self.twospheres)
        hbox0a.addStretch(1)

        #xcontroller
        hbox0 = QtGui.QHBoxLayout()
        hbox0.addWidget(x)
        vbox0 = QtGui.QVBoxLayout()
        vbox0.addWidget(self.lcd)
        vbox0.addWidget(self.sld)
        hbox0.addLayout(vbox0)

        #ycontroller
        hbox1 = QtGui.QHBoxLayout()
        hbox1.addWidget(y)
        vbox1 = QtGui.QVBoxLayout()
        vbox1.addWidget(self.lcd2)
        vbox1.addWidget(self.sld2)
        hbox1.addLayout(vbox1)

        #zcontrol
        hbox2 = QtGui.QHBoxLayout()
        hbox2.addWidget(z)
        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(self.lcd3)
        vbox2.addWidget(self.sld3)
        hbox2.addLayout(vbox2)

        #radius control
        hbox3 = QtGui.QHBoxLayout()
        hbox3.addStretch(1)
        hbox3.addWidget(radius)
        vbox3 = QtGui.QVBoxLayout()
        vbox3.addWidget(self.lcd4)
        vbox3.addWidget(self.sld4)
        hbox3.addLayout(vbox3)
        hbox3.addStretch(1)

        #index control
        hbox4 = QtGui.QHBoxLayout()
        hbox4.addStretch(1)
        hbox4.addWidget(index)
        vbox4 = QtGui.QVBoxLayout()
        vbox4.addWidget(self.lcd5)
        vbox4.addWidget(self.sld5)
        hbox4.addLayout(vbox4)
        hbox4.addStretch(1)

        hbox5 = QtGui.QHBoxLayout()
        hbox5.addWidget(self.gpu)
        hbox5.addWidget(self.cpu)
        hbox5.addWidget(self.multisphere)
        hbox5.addWidget(self.reconstruct)

        #xcontroller
        hbox7 = QtGui.QHBoxLayout()
        hbox7.addWidget(self.x2)
        vbox7 = QtGui.QVBoxLayout()
        vbox7.addWidget(self.lcd7)
        vbox7.addWidget(self.sld7)
        hbox7.addLayout(vbox7)

        #ycontroller
        hbox8 = QtGui.QHBoxLayout()
        hbox8.addWidget(self.y2)
        vbox8 = QtGui.QVBoxLayout()
        vbox8.addWidget(self.lcd8)
        vbox8.addWidget(self.sld8)
        hbox8.addLayout(vbox8)

        #zcontrol
        hbox9 = QtGui.QHBoxLayout()
        hbox9.addWidget(self.z2)
        vbox9 = QtGui.QVBoxLayout()
        vbox9.addWidget(self.lcd9)
        vbox9.addWidget(self.sld9)
        hbox9.addLayout(vbox9)

        #timer
        hbox6 = QtGui.QHBoxLayout()
        hbox6.addStretch(1)
        hbox6.addWidget(self.timer)

        #groups the matching parameters for each sphere together
        hbox_xs = QtGui.QHBoxLayout()
        hbox_xs.addLayout(hbox0)
        hbox_xs.addLayout(hbox7)
        hbox_ys = QtGui.QHBoxLayout()
        hbox_ys.addLayout(hbox1)
        hbox_ys.addLayout(hbox8)
        hbox_zs = QtGui.QHBoxLayout()
        hbox_zs.addLayout(hbox2)
        hbox_zs.addLayout(hbox9)

        #put all the parameter adjustment buttons together
        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hbox0a)
        vbox.addLayout(hbox_xs)
        vbox.addLayout(hbox_ys)
        vbox.addLayout(hbox_zs)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)
        vbox.addLayout(hbox6)
        vbox.addStretch(1)

        contentbox = QtGui.QHBoxLayout()
        contentbox.addWidget(self.label) #hologram image
        contentbox.addLayout(vbox)

        textbox = QtGui.QVBoxLayout()
        textbox.addWidget(spheretitle)
        textbox.addWidget(self.sphObject)
        textbox.addStretch(1)
        textbox.addWidget(self.schemaObject)

        schemacontrol.addWidget(width)
        schemacontrol.addWidget(self.widthText)
        schemacontrol.addWidget(height)
        schemacontrol.addWidget(self.heightText)
        schemacontrol.addWidget(wave)
        schemacontrol.addWidget(self.waveText)
        schemacontrol.addWidget(mindex)
        schemacontrol.addWidget(self.mindexText)
        schemacontrol.addWidget(pxsize)
        schemacontrol.addWidget(self.pxsizeText)
        schemacontrol.addStretch(1)

        largevbox = QtGui.QVBoxLayout()
        largevbox.addLayout(contentbox)
        largevbox.addLayout(schemacontrol)
        largevbox.addStretch(1)
        largevbox.addLayout(textbox)
        
        self.setLayout(largevbox)
        self.setWindowTitle('Interactive Hologram')    
        self.resetLayout()
        self.show()

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+q"), self, self.close)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+w"), self, self.close)


    def resetLayout(self):
        #sets geometry of window and shows/hides buttons for 
        #one or two spheres
        if self.onesphere.checkState()==2:
            self.gpu.toggle()
            self.resize(1000,700)
            self.sld7.hide()
            self.lcd7.hide()
            self.sld8.hide()
            self.lcd8.hide()
            self.sld9.hide()
            self.lcd9.hide()
            self.x2.hide()
            self.y2.hide()
            self.z2.hide()
            self.multisphere.hide()

        else:
            self.gpu.toggle()
            self.resize(1200,700)
            self.sld7.show()
            self.lcd7.show()
            self.sld8.show()
            self.lcd8.show()
            self.sld9.show()
            self.lcd9.show()
            self.x2.show()
            self.y2.show()
            self.z2.show()
            self.multisphere.show()


    def setuprecon(self):
        self.lastZ = self.lcd3.value()
        self.lastholo = self.holo


    def activate(self):
        #will want to choose reconstruction sometimes
        if self.reconstruct.isChecked():
            print 'reconstructing from z='+str(self.lastZ)
            self.lastZ = self.lcd3.value()
        else:
            self.gpu.toggle()


    def inmedChange(self):
        inmedwave = float(self.waveText.text())/float(self.mindexText.text())
        #radius
        print inmedwave
        radiuslow = 0.25*inmedwave
        radiushigh = 3.0*inmedwave
        #get current radius
        radius = self.lcd4.value()
        #reposition slider to maintain starting value
        self.sld4.setSliderPosition((radius-radiuslow)/(radiushigh-radiuslow)*100)


    def lengthscaleChange(self):
        #z from twice the radius to 60 radii
        zlow = round(2*self.lcd4.value(),1)
        zhigh = zlow+round(self.maxz_diameters*self.lcd4.value(),1)

        #get current z
        z = self.lcd3.value()

        self.sld3.setMinimum(0)
        self.sld3.setMaximum((zhigh-zlow)*10)

        #reposition slider to maintain starting value
        self.sld3.setSliderPosition(round((z-zlow)/(zhigh-zlow)*self.sld3.maximum(),0))

        #get current z
        z = self.lcd9.value()

        self.sld9.setMinimum(0)
        self.sld9.setMaximum((zhigh-zlow)*10)

        #reposition slider to maintain starting value
        self.sld9.setSliderPosition(round((z-zlow)/(zhigh-zlow)*self.sld9.maximum(),0))

        if self.lcd5.value()< float(self.mindexText.text()):
            self.lcd5.setStyleSheet('background-color: gray; color: white')
        else:
            self.lcd5.setStyleSheet('color: black')


    def updateLCD(self, value): #convert integer value scroll bars to meaningful units in LCD's
        scale = float(self.pxsizeText.text())
        sender = self.sender()

        if sender == self.sld:
            self.lcd.display(round(value*scale,2))
            #self.lcd.display(round(value*scale*float(self.heightText.text())/256.,2))

        if sender == self.sld2:
            self.lcd2.display(round(value*scale,2))
        if sender == self.sld3:
            self.lcd3.display(value/(1.0*self.sld3.maximum())*(round(self.maxz_diameters*self.lcd4.value(),1))+round(2*self.lcd4.value(),1))#z

        if sender == self.sld7:
            self.lcd7.display(round(value*scale,2))

        if sender == self.sld8:
            self.lcd8.display(round(value*scale,2))

        if sender == self.sld9:
            self.lcd9.display(value/(1.0*self.sld9.maximum())*(round(self.maxz_diameters*self.lcd4.value(),1))+round(2*self.lcd4.value(),1))#z

        if sender == self.sld4:
            inmedwave = float(self.waveText.text())/float(self.mindexText.text())
            radiuslow = .25*inmedwave
            radiushigh = 3*inmedwave
            self.lcd4.display(round((value/100.0)*(radiushigh-radiuslow)+radiuslow,2)) #r

        if sender == self.sld5: #index of refraction
            self.lcd5.display(round(value/200.0*2.0+1.0,2)) #index
            if self.lcd5.value()< float(self.mindexText.text()):
                self.lcd5.setStyleSheet('background-color: gray; color: white')
            else:
                self.lcd5.setStyleSheet('color: black')


    def rangeChange(self, value):
        if float(self.pxsizeText.text())>0:

            self.lcd.display(self.sld.value()*float(self.pxsizeText.text()))
            self.lcd2.display(self.sld2.value()*float(self.pxsizeText.text()))
            self.sld.setSliderPosition((self.lcd.value()/float(self.pxsizeText.text())))
            self.sld2.setSliderPosition((self.lcd2.value()/float(self.pxsizeText.text())))

            self.lcd7.display(self.sld7.value()*float(self.pxsizeText.text()))
            self.lcd8.display(self.sld8.value()*float(self.pxsizeText.text()))
            self.sld7.setSliderPosition((self.lcd7.value()/float(self.pxsizeText.text())))
            self.sld8.setSliderPosition((self.lcd8.value()/float(self.pxsizeText.text())))


    def rangeChange2(self, value):
        self.sld.setMaximum(float(self.heightText.text()))
        self.sld2.setMaximum(float(self.widthText.text()))

        #sender = self.sender()
        if sender == self.heightText and self.lcd.value() < float(self.heightText.text())*float(self.pxsizeText.text()):
                self.sld.setSliderPosition((self.lcd.value()/float(self.pxsizeText.text())))

        if sender == self.widthText and self.lcd2.value() < float(self.widthText.text())*float(self.pxsizeText.text()):
                self.sld2.setSliderPosition((self.lcd2.value()/float(self.pxsizeText.text())))


    def slideZ(self, sphere, schema): 
        #using reconstructions-- better to use electric field?
        '''
        When z is changed, instead of recomputing the hologram, we
        use the shortcut of reconstructing the last computed hologram.
        '''
        if sphere.parameters == self.oldReconParameters and repr(schema) == repr(self.oldReconSchema):
            self.holo = self.oldReconHolo
        else:
            source = self.sender()

            start = time.time()

            if self.lastZ == self.lcd3.value():
                self.holo = self.lastholo

            if self.lastZ < self.lcd3.value():
                self.holo = self.lastholo
                self.warning.setText('Cannot reconstruct to larger z')

            if self.lastZ > self.lcd3.value(): #reconstructing a plane between hologram and object
                self.holo = np.abs(propagate(self.lastholo, -self.lcd3.value()+self.lastZ))
                self.warning.setText('Reconstruction: hologram is approximate')

            end = time.time()
            self.oldReconHolo = self.holo
            self.oldReconParameters = sphere.parameters
            self.oldReconSchema = schema


    def calculateHologram(self): #calculate hologram with current settings
        #schema is generic to all scattering theories
        schema = ImageSchema(shape = [int(self.heightText.text()),int(self.widthText.text())], spacing = float(self.pxsizeText.text()),
            optics = Optics(wavelen = float(self.waveText.text()), 
            index = float(self.mindexText.text()), polarization = [1.0,0.0]))
        self.schemaObject.setText(str(repr(schema)))

        #fist sphere is general for both single sphere and two sphere cases
        sphere = Sphere(n = self.lcd5.value()+.0001j, 
            r = self.lcd4.value(), 
            center = (self.lcd.value(),self.lcd2.value(),self.lcd3.value()))

        if self.reconstruct.isChecked():
            self.slideZ(sphere, schema)
            start = end = 0 #fake time since reconstruction time is odd to keep track of here

        else:
            scale = self.scale
            sender = self.sender()

            start = time.time()

            if self.onesphere.checkState() == 2:
                self.sphObject.setText(repr(sphere))

            #we have two theories to choose from for computing single sphere holograms
                if sphere.parameters != self.oldscattererparameters:
                    #when changing parameters, always use GPU because it's fast enough
                    self.gpu.toggle()

                if self.cpu.isChecked():
                    if sphere.parameters == self.oldMieCPUparameters and repr(schema) == repr(self.oldMieCPUschema):
                        self.holo = self.oldMieCPUHolo
                        print 'loaded cached mie hologram'
                    else:
                        self.holo = Mie.calc_holo(sphere,schema)
                        print 'freshly calculated mie hologram'
                        self.oldMieCPUHolo = self.holo
                        self.oldMieCPUparameters = sphere.parameters
                        self.oldMieCPUschema = schema

                else:
                    if sphere.parameters == self.oldscattererparameters and repr(schema) == repr(self.oldMieGPUschema):
                        self.holo = self.oldMieGPUHolo
                        print 'loaded cached GPU hologram'
                    else:
                        self.holo = gpuMie.calc_holo(sphere, schema)
                        self.oldMieGPUHolo = self.holo                        
                        self.oldscattererparameters = sphere.parameters
                        self.oldMieGPUschema = schema
                        self.oldscatterer = sphere
                        self.oldschema = schema

            else:
            #we have three theories to choose from for computing single sphere holograms
                s1 = sphere
                s2 = Sphere(n = self.lcd5.value()+.0001j, 
                    r = self.lcd4.value(), 
                    center = (self.lcd7.value(),self.lcd8.value(),self.lcd9.value()))

                cluster = Spheres([s1, s2])
                self.sphObject.setText(repr(cluster))

                if cluster.parameters != self.oldscatterer.parameters:
                    self.gpu.toggle()

                if self.cpu.isChecked():
                    if cluster.parameters == self.oldMieCPUparameters and repr(schema) == repr(self.oldMieCPUschema):
                        self.holo = self.oldMieCPUHolo
                        print 'loaded cached CPU superposition hologram'
                    else:
                        self.holo = Mie.calc_holo(cluster, schema)
                        self.oldMieCPUHolo = self.holo
                        self.oldMieCPUschema = schema
                        self.oldMieCPUparameters = cluster.parameters

                if self.multisphere.isChecked():
                    if cluster.parameters == self.oldMultiparameters and repr(schema) == repr(self.oldMultischema):
                        self.holo = self.oldMultiHolo
                        print 'loaded cached CPU multisphere hologram'
                    else:
                        self.holo = Multisphere.calc_holo(cluster, schema)
                        self.oldtheory = 'multi'
                        self.oldMultiHolo = self.holo
                        self.oldMultiparameters = cluster.parameters
                        self.oldMultischema = schema
                        print 'multisphere calculation done'

                if self.gpu.isChecked():
                    if cluster.parameters == self.oldscattererparameters and repr(schema) == repr(self.oldschema):
                        self.holo = self.oldMieGPUHolo
                        print 'loaded cached GPU hologram'
                    else:
                        self.holo = gpuMie.calc_holo(cluster, schema)
                        self.oldMieGPUHolo = self.holo
                        self.oldschema = schema
                        self.oldscatterer = cluster
                        self.oldscattererparameters = cluster.parameters

            end = time.time()

        #display the hologram
        im = scipy.misc.toimage(self.holo) #PIL image

        #https://github.com/shuge/Enjoy-Qt-Python-Binding/blob/master/image/display_img/pil_to_qpixmap.py
        if im.mode == "RGB":
            pass
        elif im.mode == "L":
            im = im.convert("RGBA")
        data = im.tostring('raw',"RGBA")
        qim = QtGui.QImage(data, float(self.widthText.text()), float(self.heightText.text()), QtGui.QImage.Format_ARGB32)
        pixmap = QtGui.QPixmap.fromImage(qim)

        #set size of displayed image with max width or height = 500 px
        if float(self.widthText.text()) == float(self.heightText.text()):
            scaledsize = [500,500]
        else:
            if float(self.widthText.text()) > float(self.heightText.text()):
                scaledsize = [500,500/float(self.widthText.text())*float(self.heightText.text())]
            else:
                scaledsize = [500/float(self.heightText.text())*float(self.widthText.text()),500]

        myScaledPixmap = pixmap.scaled(QtCore.QSize(scaledsize[0],scaledsize[1]))

        self.warning.setText('')
        self.label.setPixmap(myScaledPixmap)

        if end-start > 0.03:
            self.timer.setText('Calc. Time: '+str(round(end-start,4))+' s')
        else:
            self.timer.setText('')
        self.timer.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter)

        if self.lcd3.value()<2*self.lcd4.value() or self.lcd9.value()<2*self.lcd4.value():
            self.warning.setText('z < sphere diameter')

        if self.lcd.value() > float(self.heightText.text())*float(self.pxsizeText.text()):
            self.warning.setText('x out of range, move slider or increase height')

        if self.lcd2.value() > float(self.widthText.text())*float(self.pxsizeText.text()):
            self.warning.setText('y out of range, move slider or increase width')

        if float(self.pxsizeText.text()) <= 0:
            self.warning.setText('pixel scale must be greater than 0')


def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Holo()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()