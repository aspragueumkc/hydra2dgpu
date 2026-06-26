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
    <field configurationFlags="NoFlag" name="inlet_type_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="inlet_count">
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
    <field configurationFlags="NoFlag" name="crest_offset">
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
    <alias index="0" field="node_id" name="Node ID"/>
    <alias index="1" field="inlet_type_id" name="Inlet Type ID"/>
    <alias index="2" field="inlet_count" name="Inlet Count"/>
    <alias index="3" field="crest_offset" name="Crest Offset"/>
    <alias index="4" field="description" name="Description"/>
  </aliases>
  <defaults>
    <default expression="" field="node_id" applyOnUpdate="0"/>
    <default expression="" field="inlet_type_id" applyOnUpdate="0"/>
    <default expression="" field="inlet_count" applyOnUpdate="0"/>
    <default expression="" field="crest_offset" applyOnUpdate="0"/>
    <default expression="" field="description" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="node_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="inlet_type_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="inlet_count" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="crest_offset" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="description" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="length(trim(&quot;node_id&quot;)) &gt; 0" field="node_id" desc=""/>
    <constraint exp="length(trim(&quot;inlet_type_id&quot;)) &gt; 0" field="inlet_type_id" desc=""/>
    <constraint exp="&quot;inlet_count&quot; IS NULL OR &quot;inlet_count&quot; &gt; 0" field="inlet_count" desc=""/>
    <constraint exp="" field="crest_offset" desc=""/>
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
    <field name="node_id"/>
    <field name="inlet_type_id"/>
    <field name="inlet_count"/>
    <field name="crest_offset"/>
    <field name="description"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
