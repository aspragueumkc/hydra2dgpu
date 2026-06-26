<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="zone_id">
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
    <field configurationFlags="NoFlag" name="n_mann">
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
    <default expression="" field="zone_id" applyOnUpdate="0"/>
    <default expression="0.035" field="n_mann" applyOnUpdate="0"/>
    <default expression="" field="priority" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="zone_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="n_mann" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="priority" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="" field="zone_id" desc=""/>
    <constraint exp="&quot;n_mann&quot; &gt;= 0" field="n_mann" desc=""/>
    <constraint exp="&quot;priority&quot; &gt;= 0" field="priority" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="zone_id"/>
    <field name="n_mann"/>
    <field name="priority"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
