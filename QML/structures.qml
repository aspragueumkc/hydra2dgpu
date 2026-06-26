<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="structure_id">
      <editWidget type="TextEdit">
        <config><Option type="Map">
          <Option value="0" name="IsMultiline" type="int"/>
          <Option value="0" name="UseHtml" type="int"/>
        </Option></config>
      </editWidget>
    </field>
    <field name="structure_type">
      <editWidget type="ValueMap">
        <config><Option type="Map">
          <Option name="map" type="List">
            <Option value="1" name="Weir" type="int"/>
            <Option value="2" name="Culvert" type="int"/>
            <Option value="3" name="Gate" type="int"/>
            <Option value="4" name="Bridge" type="int"/>
            <Option value="5" name="Pump" type="int"/>
          </Option>
        </Option></config>
      </editWidget>
    </field>
    <field name="enabled">
      <editWidget type="ValueMap">
        <config><Option type="Map">
          <Option name="map" type="List">
            <Option value="1" name="Yes" type="int"/>
            <Option value="0" name="No" type="int"/>
          </Option>
        </Option></config>
      </editWidget>
    </field>
    <field name="culvert_code">
      <editWidget type="ValueMap">
        <config><Option type="Map">
          <Option name="map" type="List">
              <value pair="— Select culvert code —">0</value>
              <value pair="Circular concrete, square edge w/ headwall">1</value>
              <value pair="Circular concrete, groove end w/ headwall">2</value>
              <value pair="Circular concrete, groove end projecting">3</value>
              <value pair="Circular concrete, mitred to slope">4</value>
              <value pair="Circular concrete, beveled ring">5</value>
              <value pair="Circular concrete, beveled ring (smoother)">6</value>
              <value pair="Circular CMP, projecting">7</value>
              <value pair="Circular CMP, projecting (different edge)">8</value>
              <value pair="Circular CMP, mitered to slope">9</value>
              <value pair="Circular CMP, mitered to slope (alt)">10</value>
              <value pair="Circular CMP, beveled end (thin wall)">11</value>
              <value pair="Circular CMP, groove end in headwall">12</value>
              <value pair="Circular CMP, groove end in headwall (alt)">13</value>
              <value pair="Circular CMP, headwall (square edge)">14</value>
              <value pair="Circular CMP, headwall (groove end)">15</value>
              <value pair="Circular CMP, headwall (thin wall projecting)">16</value>
              <value pair="Rectangular box, 30-75deg wingwall flares">17</value>
              <value pair="Rectangular box, 90deg headwall w/ chamfers">18</value>
              <value pair="Rectangular box, 0deg wingwall flares">19</value>
              <value pair="Rectangular box, 45deg wingwall flares">20</value>
              <value pair="Rectangular box, 18-33deg wingwall flares">21</value>
              <value pair="Rectangular box, 0deg wingwall flares (thick)">22</value>
              <value pair="Rectangular box, 30deg wingwall flares (thick)">23</value>
              <value pair="Rectangular box, 45deg wingwall flares (thick)">24</value>
              <value pair="Rectangular box, 0deg wingwall flares (thick alt)">25</value>
              <value pair="Rectangular box, beveled edge (1:1)">26</value>
              <value pair="Circular concrete, square edge w/ headwall (form-1 alt)">27</value>
              <value pair="Circular concrete, groove end w/ headwall (form-1 alt)">28</value>
              <value pair="Circular concrete, groove end projecting (form-1 alt)">29</value>
              <value pair="Circular CMP, projecting (form-1 alt)">30</value>
              <value pair="Circular CMP, mitered to slope (form-1 alt)">31</value>
              <value pair="Circular CMP, beveled end thin wall (form-1 alt)">32</value>
              <value pair="Circular CMP, groove end in headwall (form-1 alt)">33</value>
              <value pair="Circular CMP, headwall square edge (form-1 alt)">34</value>
              <value pair="Circular CMP, headwall groove end (form-1 alt)">35</value>
              <value pair="Circular CMP, beveled ring (form-1 alt)">36</value>
              <value pair="Circular CMP, beveled ring thick (form-1 alt)">37</value>
              <value pair="Circular concrete, beveled ring (form-1 alt)">38</value>
              <value pair="Circular pipe, beveled ring (thin wall)">39</value>
              <value pair="Circular pipe, beveled ring (thick wall)">40</value>
              <value pair="Circular pipe, 45deg beveled ring">41</value>
              <value pair="Circular pipe, 33.7deg beveled ring">42</value>
              <value pair="Circular pipe, 45deg bevel (offset)">43</value>
              <value pair="Circular pipe, 33.7deg bevel (offset)">44</value>
              <value pair="Circular CMP, prefab end section (safety)">45</value>
              <value pair="Circular CMP, prefab end section (alt)">46</value>
              <value pair="Arch CMP, 2-3-1 fill (soffit thickness 0.0625)">47</value>
              <value pair="Arch CMP, 2-3-1 fill (soffit varying)">48</value>
              <value pair="Arch CMP, 2-3-1 fill projecting (soffit varying)">49</value>
              <value pair="Arch CMP, 2-2-1 fill (soffit thickness 0.0625)">50</value>
              <value pair="Pipe arch CMP, 0.75x0.75 fill (soffit thickness 0.0625)">51</value>
              <value pair="Pipe arch CMP, 0.75x0.75 fill projecting">52</value>
              <value pair="Pipe arch CMP, 0.75x0.75 fill (soffit varying)">53</value>
              <value pair="Horizontal ellipse, concrete (form-2)">54</value>
              <value pair="Horizontal ellipse, corrugated metal (form-2)">55</value>
              <value pair="Arch CMP, 2-3-1 fill premium (form-2)">56</value>
              <value pair="Horizontal ellipse, special shape (form-2)">57</value>
          </Option>
        </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <defaults>
    <default field="structure_type" applyOnUpdate="0" expression="1"/>
    <default field="crest_elev" applyOnUpdate="0" expression="0.0"/>
    <default field="enabled" applyOnUpdate="0" expression="1"/>
    <default field="roughness_n" applyOnUpdate="0" expression="0.035"/>
    <default field="length" applyOnUpdate="0" expression="30.0"/>
    <default field="entrance_loss_k" applyOnUpdate="0" expression="0.5"/>
    <default field="exit_loss_k" applyOnUpdate="0" expression="1.0"/>
    <default field="culvert_barrels" applyOnUpdate="0" expression="1"/>
  </defaults>
  <editorlayout>tablayout</editorlayout>
  <attributeEditorForm>
    <attributeEditorContainer name="Weir" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 1" groupBox="0">
      <attributeEditorField index="-1" name="width"/>
      <attributeEditorField index="-1" name="embankment_enabled"/>
      <attributeEditorField index="-1" name="embankment_crest_elev"/>
      <attributeEditorField index="-1" name="embankment_overflow_width"/>
      <attributeEditorField index="-1" name="embankment_weir_coeff"/>
    </attributeEditorContainer>
    <attributeEditorContainer name="Culvert" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 2" groupBox="0">
      <attributeEditorField index="-1" name="culvert_shape"/>
      <attributeEditorField index="-1" name="culvert_code"/>
      <attributeEditorField index="-1" name="culvert_rise"/>
      <attributeEditorField index="-1" name="culvert_span"/>
      <attributeEditorField index="-1" name="culvert_area_m2"/>
      <attributeEditorField index="-1" name="culvert_barrels"/>
      <attributeEditorField index="-1" name="culvert_slope"/>
      <attributeEditorField index="-1" name="diameter"/>
      <attributeEditorField index="-1" name="length"/>
      <attributeEditorField index="-1" name="roughness_n"/>
      <attributeEditorField index="-1" name="inlet_invert_elev"/>
      <attributeEditorField index="-1" name="outlet_invert_elev"/>
      <attributeEditorField index="-1" name="entrance_loss_k"/>
      <attributeEditorField index="-1" name="exit_loss_k"/>
      <attributeEditorField index="-1" name="embankment_enabled"/>
      <attributeEditorField index="-1" name="embankment_crest_elev"/>
      <attributeEditorField index="-1" name="embankment_overflow_width"/>
      <attributeEditorField index="-1" name="embankment_weir_coeff"/>
    </attributeEditorContainer>
    <attributeEditorContainer name="Gate" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 3" groupBox="0">
      <attributeEditorField index="-1" name="width"/>
      <attributeEditorField index="-1" name="height"/>
      <attributeEditorField index="-1" name="opening"/>
    </attributeEditorContainer>
    <attributeEditorContainer name="Bridge" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 4" groupBox="0">
      <attributeEditorField index="-1" name="width"/>
      <attributeEditorField index="-1" name="length"/>
      <attributeEditorField index="-1" name="deck_soffit_elev"/>
      <attributeEditorField index="-1" name="deck_top_elev"/>
      <attributeEditorField index="-1" name="model_top_elev"/>
      <attributeEditorField index="-1" name="under_layers"/>
      <attributeEditorField index="-1" name="over_layers"/>
      <attributeEditorField index="-1" name="inlet_loss_k"/>
      <attributeEditorField index="-1" name="outlet_loss_k"/>
      <attributeEditorField index="-1" name="pier_count"/>
      <attributeEditorField index="-1" name="pier_width"/>
      <attributeEditorField index="-1" name="face_flux_depth_safety"/>
    </attributeEditorContainer>
    <attributeEditorContainer name="Pump" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = 5" groupBox="0">
      <attributeEditorField index="-1" name="q_pump"/>
      <attributeEditorField index="-1" name="max_flow"/>
      <attributeEditorField index="-1" name="min_head_diff"/>
      <attributeEditorField index="-1" name="max_head_diff"/>
    </attributeEditorContainer>
  </attributeEditorForm>
</qgis>