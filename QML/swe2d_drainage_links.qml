<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="link_id">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="from_node">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="to_node">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="link_type">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="conduit" name="Conduit" type="string"/>
            <Option value="lateral_simple" name="Short lateral (simplified)" type="string"/>
            <Option value="pump" name="Pump" type="string"/>
            <Option value="weir" name="Weir" type="string"/>
            <Option value="orifice" name="Orifice" type="string"/>
            <Option value="culvert" name="Culvert (HDS-5)" type="string"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="link_shape">
      <editWidget type="ValueMap">
          <config><Option type="Map">
            <Option name="map" type="List">
            <Option value="circular" name="Circular" type="string"/>
            <Option value="box" name="Box" type="string"/>
            <Option value="pipe_arch" name="Pipe arch" type="string"/>
            <Option value="custom" name="Custom area" type="string"/>
            </Option>
          </Option></config>
      </editWidget>
    </field>
    <field name="length">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="roughness_n">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="diameter">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="span">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="rise">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="area_m2">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="equiv_diameter_m">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="max_flow">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="cd">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_shape">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_code">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="0" name="Min" type="double"/>
            <Option value="57" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_rise">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_span">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_area_m2">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_barrels">
      <editWidget type="Range">
          <config><Option type="Map">
            <Option value="1" name="Min" type="double"/>
            <Option value="10" name="Max" type="double"/>
            <Option value="1" name="Step" type="double"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="culvert_slope">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="inlet_invert_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="outlet_invert_elev">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="entrance_loss_k">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="exit_loss_k">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="inlet_loss_k">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
    <field name="outlet_loss_k">
      <editWidget type="TextEdit">
          <config><Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
    <alias index="0" name="Link ID"/>
    <alias index="1" name="From Node"/>
    <alias index="2" name="To Node"/>
    <alias index="3" name="Link Type"/>
    <alias index="4" name="Link Shape"/>
    <alias index="5" name="Length"/>
    <alias index="6" name="Roughness n"/>
    <alias index="7" name="Diameter"/>
    <alias index="8" name="Span"/>
    <alias index="9" name="Rise"/>
    <alias index="10" name="Area (m²)"/>
    <alias index="11" name="Equivalent Diameter"/>
    <alias index="12" name="Max Flow"/>
    <alias index="13" name="Discharge Coefficient"/>
    <alias index="14" name="Culvert Shape"/>
    <alias index="15" name="Culvert Code"/>
    <alias index="16" name="Culvert Rise"/>
    <alias index="17" name="Culvert Span"/>
    <alias index="18" name="Culvert Area"/>
    <alias index="19" name="Culvert Barrels"/>
    <alias index="20" name="Culvert Slope"/>
    <alias index="21" name="Inlet Invert Elev."/>
    <alias index="22" name="Outlet Invert Elev."/>
    <alias index="23" name="Entrance Loss K"/>
    <alias index="24" name="Exit Loss K"/>
    <alias index="25" name="Inlet Loss K"/>
    <alias index="26" name="Outlet Loss K"/>
  </aliases>
  <defaults>
  </defaults>
  <constraintExpressions>
      <constraint exp="length(trim(&quot;link_id&quot;)) &gt; 0" field="link_id" desc=""/>
      <constraint exp="length(trim(&quot;from_node&quot;)) &gt; 0" field="from_node" desc=""/>
      <constraint exp="length(trim(&quot;to_node&quot;)) &gt; 0" field="to_node" desc=""/>
      <constraint exp="&quot;link_type&quot; IN ('conduit','lateral_simple','pump','weir','orifice','culvert')" field="link_type" desc=""/>
      <constraint exp="&quot;link_shape&quot; IS NULL OR &quot;link_shape&quot; IN ('circular','box','pipe_arch','custom')" field="link_shape" desc=""/>
      <constraint exp="&quot;length&quot; IS NULL OR &quot;length&quot; &gt; 0" field="length" desc=""/>
      <constraint exp="&quot;roughness_n&quot; IS NULL OR &quot;roughness_n&quot; &gt; 0" field="roughness_n" desc=""/>
      <constraint exp="&quot;diameter&quot; IS NULL OR &quot;diameter&quot; &gt; 0" field="diameter" desc=""/>
      <constraint exp="&quot;span&quot; IS NULL OR &quot;span&quot; &gt; 0" field="span" desc=""/>
      <constraint exp="&quot;rise&quot; IS NULL OR &quot;rise&quot; &gt; 0" field="rise" desc=""/>
      <constraint exp="&quot;area_m2&quot; IS NULL OR &quot;area_m2&quot; &gt; 0" field="area_m2" desc=""/>
  </constraintExpressions>
  <editform></editform>
  <editforminit></editforminit>
  <editforminitcodesource>0</editforminitcodesource>
  <editforminitfilepath></editforminitfilepath>
  <editforminitcode><![CDATA[]]></editforminitcode>
  <featformsuppress>0</featformsuppress>
  <editorlayout>tablayout</editorlayout>
  <editable>
    <field name="link_id"/>
    <field name="from_node"/>
    <field name="to_node"/>
    <field name="link_type"/>
    <field name="link_shape"/>
    <field name="length"/>
    <field name="roughness_n"/>
    <field name="diameter"/>
    <field name="span"/>
    <field name="rise"/>
    <field name="area_m2"/>
    <field name="equiv_diameter_m"/>
    <field name="max_flow"/>
    <field name="cd"/>
    <field name="culvert_shape"/>
    <field name="culvert_code"/>
    <field name="culvert_rise"/>
    <field name="culvert_span"/>
    <field name="culvert_area_m2"/>
    <field name="culvert_barrels"/>
    <field name="culvert_slope"/>
    <field name="inlet_invert_elev"/>
    <field name="outlet_invert_elev"/>
    <field name="entrance_loss_k"/>
    <field name="exit_loss_k"/>
    <field name="inlet_loss_k"/>
    <field name="outlet_loss_k"/>
  </editable>
  <labelOnTop>
  </labelOnTop>
  <reuseLastValue>
  </reuseLastValue>
  <dataDefinedFieldProperties/>
  <widgets/>
</qgis>