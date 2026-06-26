<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="node_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="invert_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="max_depth">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="rim_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="crest_elev">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="node_type">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="junction" name="Junction" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="outfall" name="Outfall" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="storage" name="Storage" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="inlet" name="Inlet" type="string"/>
              </Option>
              <Option type="Map">
                <Option value="pipe_end" name="Pipe end" type="string"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="surface_area">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="outfall_area">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="zero_storage">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="0" name="No" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="1" name="Yes" type="int"/>
              </Option>
              </Option>
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
    <alias index="5" field="" name=""/>
    <alias index="6" field="" name=""/>
    <alias index="7" field="" name=""/>
    <alias index="8" field="" name=""/>
  </aliases>
  <defaults>
    <default expression="" field="node_id" applyOnUpdate="0"/>
    <default expression="" field="invert_elev" applyOnUpdate="0"/>
    <default expression="" field="max_depth" applyOnUpdate="0"/>
    <default expression="" field="rim_elev" applyOnUpdate="0"/>
    <default expression="" field="crest_elev" applyOnUpdate="0"/>
    <default expression="" field="node_type" applyOnUpdate="0"/>
    <default expression="" field="surface_area" applyOnUpdate="0"/>
    <default expression="" field="outfall_area" applyOnUpdate="0"/>
    <default expression="" field="zero_storage" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="node_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="invert_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="max_depth" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="rim_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="crest_elev" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="node_type" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="surface_area" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="outfall_area" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="zero_storage" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="length(trim(&quot;node_id&quot;)) &gt; 0" field="node_id" desc=""/>
    <constraint exp="" field="invert_elev" desc=""/>
    <constraint exp="&quot;max_depth&quot; IS NULL OR &quot;max_depth&quot; &gt; 0" field="max_depth" desc=""/>
    <constraint exp="&quot;rim_elev&quot; IS NULL OR &quot;rim_elev&quot; &gt;= &quot;invert_elev&quot;" field="rim_elev" desc=""/>
    <constraint exp="&quot;crest_elev&quot; IS NULL OR &quot;crest_elev&quot; &gt;= &quot;invert_elev&quot;" field="crest_elev" desc=""/>
    <constraint exp="&quot;node_type&quot; IN ('junction','outfall','storage','inlet','pipe_end')" field="node_type" desc=""/>
    <constraint exp="&quot;surface_area&quot; IS NULL OR &quot;surface_area&quot; &gt; 0" field="surface_area" desc=""/>
    <constraint exp="&quot;outfall_area&quot; IS NULL OR &quot;outfall_area&quot; &gt; 0" field="outfall_area" desc=""/>
    <constraint exp="&quot;zero_storage&quot; IS NULL OR &quot;zero_storage&quot; IN (0,1)" field="zero_storage" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="node_id"/>
    <field name="invert_elev"/>
    <field name="max_depth"/>
    <field name="rim_elev"/>
    <field name="crest_elev"/>
    <field name="node_type"/>
    <field name="surface_area"/>
    <field name="outfall_area"/>
    <field name="zero_storage"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
