<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="gage_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="name">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="hyetograph_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
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
    <field configurationFlags="NoFlag" name="priority">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="100" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
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
  </aliases>
  <defaults>
    <default expression="" field="gage_id" applyOnUpdate="0"/>
    <default expression="" field="name" applyOnUpdate="0"/>
    <default expression="" field="hyetograph_id" applyOnUpdate="0"/>
    <default expression="" field="units" applyOnUpdate="0"/>
    <default expression="" field="priority" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="gage_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="name" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="hyetograph_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="units" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="priority" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="length(trim(&quot;gage_id&quot;)) &gt; 0" field="gage_id" desc=""/>
    <constraint exp="" field="name" desc=""/>
    <constraint exp="length(trim(&quot;hyetograph_id&quot;)) &gt; 0" field="hyetograph_id" desc=""/>
    <constraint exp="&quot;units&quot; IS NULL OR &quot;units&quot; IN ('mm/hr','in/hr','mm','in')" field="units" desc=""/>
    <constraint exp="" field="priority" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="gage_id"/>
    <field name="name"/>
    <field name="hyetograph_id"/>
    <field name="units"/>
    <field name="priority"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
