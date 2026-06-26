<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="region_id">
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
    <field configurationFlags="NoFlag" name="target_size">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="cell_type">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="triangular" name="Triangular" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="quadrilateral" name="Quadrilateral" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="cartesian" name="Cartesian" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="channel_generator" name="Channel generator" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="empty" name="Empty" type="string"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="edge_len_1">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="edge_len_2">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="edge_len_3">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="edge_len_4">
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
    <alias index="6" field="" name=""/>
  </aliases>
  <defaults>
    <default expression="" field="region_id" applyOnUpdate="0"/>
    <default expression="" field="target_size" applyOnUpdate="0"/>
    <default expression="" field="cell_type" applyOnUpdate="0"/>
    <default expression="" field="edge_len_1" applyOnUpdate="0"/>
    <default expression="" field="edge_len_2" applyOnUpdate="0"/>
    <default expression="" field="edge_len_3" applyOnUpdate="0"/>
    <default expression="" field="edge_len_4" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="region_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="target_size" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="cell_type" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="edge_len_1" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="edge_len_2" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="edge_len_3" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="edge_len_4" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="" field="region_id" desc=""/>
    <constraint exp="&quot;target_size&quot; &gt; 0" field="target_size" desc=""/>
    <constraint exp="&quot;cell_type&quot; IN ('triangular','quadrilateral','cartesian','channel_generator','empty')" field="cell_type" desc=""/>
    <constraint exp="&quot;edge_len_1&quot; IS NULL OR &quot;edge_len_1&quot; &gt; 0" field="edge_len_1" desc=""/>
    <constraint exp="&quot;edge_len_2&quot; IS NULL OR &quot;edge_len_2&quot; &gt; 0" field="edge_len_2" desc=""/>
    <constraint exp="&quot;edge_len_3&quot; IS NULL OR &quot;edge_len_3&quot; &gt; 0" field="edge_len_3" desc=""/>
    <constraint exp="&quot;edge_len_4&quot; IS NULL OR &quot;edge_len_4&quot; &gt; 0" field="edge_len_4" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="region_id"/>
    <field name="target_size"/>
    <field name="cell_type"/>
    <field name="edge_len_1"/>
    <field name="edge_len_2"/>
    <field name="edge_len_3"/>
    <field name="edge_len_4"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
