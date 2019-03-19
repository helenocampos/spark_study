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
public class MeasurementModel implements Serializable
{
    private float hora;
    private int temperatura;

    public MeasurementModel(float hora, int temperatura)
    {
        this.hora = hora;
        this.temperatura = temperatura;
    }
    
    public float getHora()
    {
        return hora;
    }

    public void setHora(float hora)
    {
        this.hora = hora;
    }

    public int getTemperatura()
    {
        return temperatura;
    }

    public void setTemperatura(int temperatura)
    {
        this.temperatura = temperatura;
    }
    
    
    
}
