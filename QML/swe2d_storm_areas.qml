<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="storm_id">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="1" name="Min" type="int"/>
              <Option value="2147483647" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
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
  </aliases>
  <defaults>
    <default expression="" field="storm_id" applyOnUpdate="0"/>
    <default expression="" field="name" applyOnUpdate="0"/>
    <default expression="" field="priority" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="storm_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="name" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="priority" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="" field="storm_id" desc=""/>
    <constraint exp="" field="name" desc=""/>
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
    <field name="storm_id"/>
    <field name="name"/>
    <field name="priority"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
