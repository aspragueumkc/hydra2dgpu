<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">
  <fieldConfiguration>
    <field configurationFlags="NoFlag" name="bc_type">
      <editWidget type="ValueMap">
          <config>
            <Option type="Map">
              <Option name="map" type="List">
              <Option type="Map">
                <Option value="1" name="Wall (zero normal flux)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="2" name="Inflow Q (total discharge)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="3" name="Stage (prescribed WSE)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="6" name="Normal Depth (prescribed depth)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="7" name="Normal Depth (friction slope Sf)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="102" name="Timeseries Flow Q" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="103" name="Timeseries Stage" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="4" name="Open (zero-gradient)" type="int"/>
              </Option>
              <Option type="Map">
                <Option value="5" name="Reflecting" type="int"/>
              </Option>
              </Option>
            </Option>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="bc_value">
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
    <field configurationFlags="NoFlag" name="hydrograph">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="hydrograph_id">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
    <field configurationFlags="NoFlag" name="hydrograph_layer">
      <editWidget type="TextEdit">
          <config>
            <Option/>
          </config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" field="bc_type" name="BC Type"/>
    <alias index="1" field="bc_value" name="BC Value"/>
    <alias index="2" field="priority" name="Priority"/>
    <alias index="3" field="hydrograph" name="Hydrograph"/>
    <alias index="4" field="hydrograph_id" name="Hydrograph ID"/>
    <alias index="5" field="hydrograph_layer" name="Hydrograph Layer"/>
  </aliases>
  <defaults>
    <default expression="" field="bc_type" applyOnUpdate="0"/>
    <default expression="" field="bc_value" applyOnUpdate="0"/>
    <default expression="" field="priority" applyOnUpdate="0"/>
    <default expression="" field="hydrograph" applyOnUpdate="0"/>
    <default expression="" field="hydrograph_id" applyOnUpdate="0"/>
    <default expression="" field="hydrograph_layer" applyOnUpdate="0"/>
  </defaults>
  <constraints>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="bc_type" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="bc_value" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="1" constraints="4" field="priority" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="hydrograph" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="hydrograph_id" unique_strength="0"/>
    <constraint notnull_strength="0" exp_strength="0" constraints="0" field="hydrograph_layer" unique_strength="0"/>
  </constraints>
  <constraintExpressions>
    <constraint exp="&quot;bc_type&quot; IN (1,2,3,4,5,6,7,102,103)" field="bc_type" desc=""/>
    <constraint exp="" field="bc_value" desc=""/>
    <constraint exp="&quot;priority&quot; &gt;= 0" field="priority" desc=""/>
    <constraint exp="" field="hydrograph" desc=""/>
    <constraint exp="" field="hydrograph_id" desc=""/>
    <constraint exp="" field="hydrograph_layer" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="bc_type"/>
    <field name="bc_value"/>
    <field name="priority"/>
    <field name="hydrograph"/>
    <field name="hydrograph_id"/>
    <field name="hydrograph_layer"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>
