<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="hydrograph_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="bc_type">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="1" name="Min" type="int"/>
              <Option value="103" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
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
  </aliases>
  <defaults>
    <default expression="" field="hydrograph_id" applyOnUpdate="0"/>
    <default expression="" field="bc_type" applyOnUpdate="0"/>
    <default expression="" field="Time" applyOnUpdate="0"/>
    <default expression="" field="Value" applyOnUpdate="0"/>
    <default expression="" field="description" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="hydrograph_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="bc_type" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="Time" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="Value" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="description" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="" field="hydrograph_id" desc=""/>
    <constraint exp="" field="bc_type" desc=""/>
    <constraint exp="" field="Time" desc=""/>
    <constraint exp="" field="Value" desc=""/>
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
    <field name="hydrograph_id"/>
    <field name="bc_type"/>
    <field name="Time"/>
    <field name="Value"/>
    <field name="description"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
