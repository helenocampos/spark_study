/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package model;

import java.io.Serializable;

/**
 *
 * @author helenocampos
 */
public class MeasurementModelLabeled extends MeasurementModel implements Serializable
{
    private int label;
    
    public MeasurementModelLabeled(float hora, int temperatura, int label)
    {
        super(hora, temperatura);
        this.label = label;
    }

    public int getLabel()
    {
        return label;
    }

    public void setLabel(int label)
    {
        this.label = label;
    }
    
    
    
}
