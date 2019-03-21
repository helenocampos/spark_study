/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package model;

import com.fasterxml.jackson.databind.ObjectMapper;
import commons.TemperatureMeasurement;
import java.util.Map;
import org.apache.kafka.common.serialization.Deserializer;

/**
 *
 * @author helenocampos
 */
public class TemperatureMeasurementDeserializer implements Deserializer<TemperatureMeasurement>
{

    @Override
    public void configure(Map<String, ?> configs, boolean isKey)
    {
    }

    @Override
    public TemperatureMeasurement deserialize(String topic, byte[] data)
    {
        ObjectMapper mapper = new ObjectMapper();
        TemperatureMeasurement object = null;
        try
        {
            object = mapper.readValue(data, TemperatureMeasurement.class);
        } catch (Exception exception)
        {
            System.out.println("Error in deserializing bytes " + exception);
        }
        return object;
    }

    @Override
    public void close()
    {
    }
}
