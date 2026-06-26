<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="line_id">
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
    <field configurationFlags="NoFlag" name="enabled">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="1" name="Yes" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="0" name="No" type="int"/>
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
    <alias index="0" field="line_id" name="Line ID"/>
    <alias index="1" field="name" name="Name"/>
    <alias index="2" field="enabled" name="Enabled"/>
    <alias index="3" field="priority" name="Priority"/>
  </aliases>
  <defaults>
    <default expression="" field="line_id" applyOnUpdate="0"/>
    <default expression="" field="name" applyOnUpdate="0"/>
    <default expression="" field="enabled" applyOnUpdate="0"/>
    <default expression="" field="priority" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="line_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="name" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="enabled" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="priority" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="&quot;line_id&quot; IS NULL OR &quot;line_id&quot; &gt;= 0" field="line_id" desc=""/>
    <constraint exp="" field="name" desc=""/>
    <constraint exp="&quot;enabled&quot; IS NULL OR &quot;enabled&quot; IN (0,1)" field="enabled" desc=""/>
    <constraint exp="&quot;priority&quot; IS NULL OR &quot;priority&quot; &gt;= 0" field="priority" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="line_id"/>
    <field name="name"/>
    <field name="enabled"/>
    <field name="priority"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
