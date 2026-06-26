<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="hyetograph_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="Time">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="Value">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="value_type">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="intensity" name="Intensity" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="incremental" name="Incremental depth" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="cumulative" name="Cumulative depth" type="string"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="units">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="mm/hr" name="mm/hr" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="in/hr" name="in/hr" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="mm" name="mm" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="in" name="in" type="string"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="description">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" field="" name=""/>
    <alias index="1" field="" name=""/>
    <alias index="2" field="" name=""/>
    <alias index="3" field="" name=""/>
    <alias index="4" field="" name=""/>
    <alias index="5" field="" name=""/>
  </aliases>
  <defaults>
    <default expression="" field="hyetograph_id" applyOnUpdate="0"/>
    <default expression="" field="Time" applyOnUpdate="0"/>
    <default expression="" field="Value" applyOnUpdate="0"/>
    <default expression="" field="value_type" applyOnUpdate="0"/>
    <default expression="" field="units" applyOnUpdate="0"/>
    <default expression="" field="description" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="hyetograph_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="Time" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="Value" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="value_type" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="units" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="description" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="length(trim(&quot;hyetograph_id&quot;)) &gt; 0" field="hyetograph_id" desc=""/>
    <constraint exp="length(trim(&quot;Time&quot;)) &gt; 0" field="Time" desc=""/>
    <constraint exp="&quot;Value&quot; &gt;= 0" field="Value" desc=""/>
    <constraint exp="&quot;value_type&quot; IS NULL OR &quot;value_type&quot; IN ('intensity','incremental','cumulative')" field="value_type" desc=""/>
    <constraint exp="&quot;units&quot; IS NULL OR &quot;units&quot; IN ('mm/hr','in/hr','mm','in')" field="units" desc=""/>
    <constraint exp="" field="description" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="hyetograph_id"/>
    <field name="Time"/>
    <field name="Value"/>
    <field name="value_type"/>
    <field name="units"/>
    <field name="description"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
