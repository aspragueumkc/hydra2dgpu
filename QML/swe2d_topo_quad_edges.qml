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
    <field configurationFlags="NoFlag" name="edge_id">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="1" name="Edge 1 (left)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="2" name="Edge 2 (right)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="3" name="Edge 3 (bottom)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="4" name="Edge 4 (top)" type="int"/>
              </Option>
              </Option>
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
    <field configurationFlags="NoFlag" name="n_layers">
      <editWidget type="Range">
          <config>
            <Option type="Map">
              <Option value="0" name="Min" type="int"/>
              <Option value="1000" name="Max" type="int"/>
              <Option value="1" name="Step" type="int"/>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="first_height">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="growth_rate">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" field="region_id" name="Region ID"/>
    <alias index="1" field="edge_id" name="Edge ID"/>
    <alias index="2" field="target_size" name="Target Size"/>
    <alias index="3" field="n_layers" name="Number of Layers"/>
    <alias index="4" field="first_height" name="First Cell Height"/>
    <alias index="5" field="growth_rate" name="Growth Rate"/>
  </aliases>
  <defaults>
    <default expression="" field="region_id" applyOnUpdate="0"/>
    <default expression="" field="edge_id" applyOnUpdate="0"/>
    <default expression="" field="target_size" applyOnUpdate="0"/>
    <default expression="" field="n_layers" applyOnUpdate="0"/>
    <default expression="" field="first_height" applyOnUpdate="0"/>
    <default expression="" field="growth_rate" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="region_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="edge_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="target_size" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="n_layers" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="first_height" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="growth_rate" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="&quot;region_id&quot; &gt;= 0" field="region_id" desc=""/>
    <constraint exp="&quot;edge_id&quot; IN (1,2,3,4)" field="edge_id" desc=""/>
    <constraint exp="&quot;target_size&quot; IS NULL OR &quot;target_size&quot; &gt; 0" field="target_size" desc=""/>
    <constraint exp="&quot;n_layers&quot; &gt;= 0" field="n_layers" desc=""/>
    <constraint exp="&quot;first_height&quot; IS NULL OR &quot;first_height&quot; &gt; 0" field="first_height" desc=""/>
    <constraint exp="&quot;growth_rate&quot; IS NULL OR &quot;growth_rate&quot; &gt; 0" field="growth_rate" desc=""/>
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
    <field name="edge_id"/>
    <field name="target_size"/>
    <field name="n_layers"/>
    <field name="first_height"/>
    <field name="growth_rate"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
