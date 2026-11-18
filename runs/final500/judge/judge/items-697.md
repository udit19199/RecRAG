Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 6 items.

=== ITEM id=49d8463be6df5e9a ===
Gold sentences and facts:
Sentence 0: It is also known as an `` immediate hemolytic transfusion reaction '' . This is a medical emergency as it results from rapid destruction of the donor red blood cells by host antibodies ( IgG , IgM ) . It is usually related to ABO blood group incompatibility - the most severe of which often involves group A red cells being given to a patient with group O type blood . Properdin then binds to complement C3 in the donor blood , facilitating the reaction through the alternate pathway cascade . The donor cells also become coated with IgG and are subsequently removed by macrophages in the reticuloendothelial system ( RES ) . Jaundice and disseminated intravascular coagulation ( DIC ) may also occur . The most common cause is clerical error ( i.e. the wrong unit of blood being given to the patient ) .
  Fact 0.0: The condition is also known as an immediate hemolytic transfusion reaction
  Fact 0.1: The reaction is a medical emergency
  Fact 0.2: The reaction results from rapid destruction of the donor red blood cells by host antibodies
  Fact 0.3: The host antibodies involved are IgG and IgM
  Fact 0.4: The reaction is usually related to ABO blood group incompatibility
  Fact 0.5: The most severe ABO incompatibility often involves group A red cells being given to a patient with group O type blood
  Fact 0.6: Properdin binds to complement C3 in the donor blood
  Fact 0.7: Properdin's binding facilitates the reaction through the alternate pathway cascade
  Fact 0.8: The donor cells become coated with IgG
  Fact 0.9: The IgG-coated donor cells are subsequently removed by macrophages in the reticuloendothelial system (RES)
  Fact 0.10: Jaundice may also occur
  Fact 0.11: Disseminated intravascular coagulation (DIC) may also occur
  Fact 0.12: The most common cause of the reaction is clerical error
  Fact 0.13: Clerical error means the wrong unit of blood being given to the patient

Retrieved items:
<item 1>
Acute hemolytic transfusion reaction
, lactated Ringer 's solution or normal saline ) , and osmotic diuresis with mannitol . If DIC is documented and bleeding requires treatment , transfusions of
frozen plasma , pooled cryoprecipitates for fibrinogen , and / or platelet concentrates may be indicated . In case of non-antibody mediated AHR This does not
require rigorous management . Diuresis induced by an infusion of normal saline until the intense red color of hemoglobinuria ceases is usually adequate
treatment . Prognosis ( edit ) The major complication is that the hemoglobin , released by the destruction of red blood cells , may cause acute kidney failure (
also known as the `` oliguric phase '' ) . About 20 deaths occur annually in the US due to AHTR .
</item 1>

<item 2>
Acute hemolytic transfusion reaction
ISBN 1 - 4051 - 3649 - 9 . Jump up ^ Molthan L , Matulewicz TJ , Bansal - Carver B , Benz EJ ( 1984 ) . `` An immediate hemolytic transfusion reaction due to
anti-C and a delayed hemolytic transfusion reaction due to anti-Ce + e : hemoglobinemia , hemoglobinuria and transient impaired renal function '' . Vox Sang. 47
( 5 ) : 348 -- 53 . doi : 10.1111 / j. 1423 - 0410.1984. tb04138. x . PMID 6438912 . Jump up ^ `` The 2011 National Blood Collection and Utilization Survey
Report '' ( PDF ) . Department of Health and Human Services . Retrieved 21 January 2016 . Jump up ^ Bolton - Maggs , PHB ; Poles , D ; et al . ( The Serious
Hazards of Transfusion ( SHOT ) Steering Group ) ( 2015 ) . The 2014 Annual SHOT Report ( 2015 ) ( PDF ) .
</item 2>

<item 3>
Acute hemolytic transfusion reaction
Acute hemolytic transfusion reaction - wikipedia Acute hemolytic transfusion reaction Jump to : navigation , search This article may be too technical for most
readers to understand . Please help improve it to make it understandable to non-experts , without removing the technical details . ( January 2018 ) ( Learn how
and when to remove this template message ) An acute hemolytic transfusion reaction ( AHTR ) is a type of transfusion reaction that is associated with hemolysis
. It occurs very soon after the transfusion , and within 24 hrs post-transfusion . It can occur quickly upon transfusing a few milliliters , or up to 1 -- 2
hours post-transfusion . It is also known as an `` immediate hemolytic transfusion reaction '' . This is a medical emergency as it results from rapid
destruction of the donor red blood cells by host antibodies ( IgG , IgM ) . It is usually related to ABO blood group incompatibility - the most severe of which
often involves group A red cells being given to a patient with group O type blood .
</item 3>

<item 4>
Acute hemolytic transfusion reaction
Types ( edit ) Acute hemolytic transfusion reactions are divided into two types . Intravascular haemolysis ( edit ) The most common cause of this reaction is
ABO blood groups incompatibility , secondary to human errors . In this process , the donor 's red blood cells are destroyed by recipient 's preformed antibodies
through the activation of complement system . Thus , the donor 's red blood cells break open and releases haemoglobin into the blood . The haemoglobin is then
excreted into urine , causes haemoglobinuria . Meanwhile , haemoglobin can also pass through the liver system and process into bilirubin , which accumulates
under the skin , causing jaundice . The release of red blood cell debris and the activation of complement system causes uncontrolled clotting ( disseminated
intravascular coagulation ) and circulatory shock . Extravascular haemolysis ( edit ) In this case , the donor 's red blood cells are recognised and bound to
IgG antibodies .
</item 4>

<item 5>
Acute hemolytic transfusion reaction
the wrong unit of blood being given to the patient ) . Contents ( hide ) 1 Epidemiology 2 Types 2.1 Intravascular haemolysis 2.2 Extravascular haemolysis 3
Signs and symptoms 4 Pathophysiology 4.1 Immune mediated 4.2 Non-immune mediated 5 Diagnosis 5.1 Differential diagnoses 6 Management 7 Prognosis 8 Epidemiology
9 References Epidemiology ( edit ) In the USA in 2011 there were 48 episodes of acute hemolysis due to a mismatch in ABO blood type ( 1 in 495,207 blood
components transfused ) and 168 episodes of hemolysis due to other causes ( 1 in 124,525 blood components transfused ) . In the UK in 2014 there were 18
episodes of hemolysis due to other causes ( 1 in 147,972 blood components transfused ) . Types ( edit ) Acute hemolytic transfusion reactions are divided into
two types .
</item 5>

<item 6>
Acute hemolytic transfusion reaction (AHTR) is a medical emergency occurring during or within 24 hours of transfusion, often after a clerical error causes
ABO-incompatible blood to be given. Recipient antibodies destroy donor red cells: complement-mediated intravascular hemolysis can cause hemoglobinuria, shock,
DIC, jaundice and kidney failure; IgG-mediated extravascular destruction is usually milder. Signs include fever, hypotension and red urine; testing may
</item 6>

<item 7>
The excerpt briefly describes acute hemolytic transfusion reaction: usually ABO incompatibility triggers red-cell destruction, potentially causing fever,
hypotension, hemoglobinuria, shock, DIC, jaundice, and kidney failure. Most of the text compiles retinal-science topics and references: anatomy, neural
pathways, OCT, vascular patterns and cardiovascular risk, disease diagnosis, gene therapy (especially AAV/RPE65 for LCA), neuroprotection, transplantation,
artificial retinas,
</item 7>

<item 8>
Acute hemolytic transfusion reaction
Macrophages then recognise these IgG antibodies and engulf these red blood cells , removing the red blood cells from the blood circulation into liver and spleen
. These red blood cells are then destroyed by macrophages in these organs . Anti-Rh antibodies most commonly directed this type of reaction . This process is
much more controlled when compared with intravascular haemolysis . Liver has adequate capacity in processing the bilirubin . Therefore , jaundice rarely occurs
. Signs and symptoms ( edit ) Early signs are fever , low blood pressure , anxiety , and red - colored urine . Late signs are generalized bleeding , caused by
disseminated intravascular coagulation , and low blood pressure . Laboratory assessment is based on a positive Direct Antiglobulin Test ( DAT ) , a decrease in
serum haptoglobin , and an increase in blood levels of the enzyme lactate dehydrogenase and indirect bilirubin levels . Pathophysiology ( edit ) Immune mediated
( edit ) This may be caused by preformed IgM anti-A , anti-B or both .
</item 8>

<item 9>
Acute hemolytic transfusion reaction
It may also be caused by other non-ABO IgG antibodies such as Rh , Kell , or Duffy . The former results in a severe intravascular hemolysis and the latter
typically causes extravascular hemolysis . The reactions are mediated by cytokines like TNF , IL - 8 , monocyte chemoattractant protein , IL - 1 etc .
Non-immune mediated ( edit ) These occur when RBCs are damaged before transfusion , resulting in hemoglobinemia and hemoglobinuria . Clinical symptoms are nil
or milder . Diagnosis ( edit ) Differential diagnoses ( edit ) Disseminated intravascular coagulation Hemolytic anemia Pulmonary edema Septic shock Management (
edit ) In case of antibody - mediated acute hemolytic transfusion reactions , Immediately discontinue the transfusion while maintaining venous access for
emergency management . Anticipate low blood pressure , kidney failure , and DIC . Prophylactic measures to reduce the risk of renal failure may include low -
dose dopamine , vigorous hydration with intravenous crystalloid solutions ( e.g.
</item 9>

<item 10>
Acute hemolytic transfusion reaction
Properdin then binds to complement C3 in the donor blood , facilitating the reaction through the alternate pathway cascade . The donor cells also become coated
with IgG and are subsequently removed by macrophages in the reticuloendothelial system ( RES ) . Jaundice and disseminated intravascular coagulation ( DIC ) may
also occur . The most common cause is clerical error ( i.e. the wrong unit of blood being given to the patient ) .
</item 10>
=== END ITEM id=49d8463be6df5e9a ===

=== ITEM id=d0feb1dc4412319e ===
Gold sentences and facts:
Sentence 0: In the light - dependent reactions ( mainly three vital reaction steps : ( I ) Hill and Bendall 's Z - scheme of cyclic and non-cyclic reactions , ( II ) Mitchell 's chemi - osmosis for ATP synthesis and ( III ) Dolai 's S - state diagrams in water - splitting ) , one molecule of the pigment chlorophyll absorbs one photon and loses one electron . This electron is passed to a modified form of chlorophyll called pheophytin , which passes the electron to a quinone molecule , starting the flow of electrons down an electron transport chain that leads to the ultimate reduction of NADP to NADPH . In addition , this creates a proton gradient ( energy gradient ) across the chloroplast membrane , which is used by ATP synthase in the synthesis of ATP . The chlorophyll molecule ultimately regains the electron it lost when a water molecule is split in a process called photolysis , which releases a dioxygen ( O ) molecule as a waste product .
  Fact 0.0: The light-dependent reactions have three vital reaction steps.
  Fact 0.1: Step I of the light-dependent reactions is Hill and Bendall's Z-scheme of cyclic and non-cyclic reactions.
  Fact 0.2: Step II of the light-dependent reactions is Mitchell's chemi-osmosis for ATP synthesis.
  Fact 0.3: Step III of the light-dependent reactions is Dolai's S-state diagrams in water-splitting.
  Fact 0.4: In the light-dependent reactions, one molecule of the pigment chlorophyll absorbs one photon.
  Fact 0.5: In the light-dependent reactions, one molecule of chlorophyll loses one electron.
  Fact 0.6: The lost electron is passed to a modified form of chlorophyll called pheophytin.
  Fact 0.7: Pheophytin passes the electron to a quinone molecule.
  Fact 0.8: This starts the flow of electrons down an electron transport chain.
  Fact 0.9: The electron transport chain leads to the ultimate reduction of NADP to NADPH.
  Fact 0.10: The electron flow creates a proton gradient (energy gradient) across the chloroplast membrane.
  Fact 0.11: ATP synthase uses the proton gradient in the synthesis of ATP.
  Fact 0.12: The chlorophyll molecule ultimately regains the electron it lost.
  Fact 0.13: The electron is regained when a water molecule is split in a process called photolysis.
  Fact 0.14: Photolysis releases a dioxygen (O) molecule as a waste product.
Sentence 1: Although photosynthesis is performed differently by different species , the process always begins when energy from light is absorbed by proteins called reaction centres that contain green chlorophyll pigments . In plants , these proteins are held inside organelles called chloroplasts , which are most abundant in leaf cells , while in bacteria they are embedded in the plasma membrane . In these light - dependent reactions , some energy is used to strip electrons from suitable substances , such as water , producing oxygen gas . The hydrogen freed by the splitting of water is used in the creation of two further compounds that act as an immediate energy storage means : reduced nicotinamide adenine dinucleotide phosphate ( NADPH ) and adenosine triphosphate ( ATP ) , the `` energy currency '' of cells .
  Fact 1.0: Photosynthesis is performed differently by different species.
  Fact 1.1: Photosynthesis always begins when energy from light is absorbed by proteins called reaction centres.
  Fact 1.2: Reaction centres are proteins that contain green chlorophyll pigments.
  Fact 1.3: In plants, reaction centres are held inside organelles called chloroplasts.
  Fact 1.4: Chloroplasts are most abundant in leaf cells.
  Fact 1.5: In bacteria, reaction centres are embedded in the plasma membrane.
  Fact 1.6: In the light-dependent reactions, some energy is used to strip electrons from suitable substances, such as water.
  Fact 1.7: The light-dependent reactions produce oxygen gas.
  Fact 1.8: The hydrogen freed by splitting water is used in the creation of two further compounds.
  Fact 1.9: The two compounds act as an immediate energy storage means.
  Fact 1.10: The two compounds are reduced nicotinamide adenine dinucleotide phosphate (NADPH) and adenosine triphosphate (ATP).
  Fact 1.11: ATP is the energy currency of cells.
Sentence 2: Photosynthesis occurs in two stages . In the first stage , light - dependent reactions or light reactions capture the energy of light and use it to make the energy - storage molecules ATP and NADPH . During the second stage , the light - independent reactions use these products to capture and reduce carbon dioxide .
  Fact 2.0: Photosynthesis occurs in two stages.
  Fact 2.1: In the first stage, light-dependent reactions (light reactions) capture the energy of light.
  Fact 2.2: In the first stage, the light-dependent reactions use the captured energy to make the energy-storage molecules ATP and NADPH.
  Fact 2.3: During the second stage, the light-independent reactions use ATP and NADPH.
  Fact 2.4: In the second stage, the light-independent reactions capture and reduce carbon dioxide.

Retrieved items:
<item 1>
Photorespiration is a light-dependent pathway that recycles products of RuBisCO’s oxygenase activity. In a simplified reaction, two glycolate molecules and ATP
are converted into 3-phosphoglycerate, carbon dioxide, ADP, and ammonia, while oxygen is consumed and carbon dioxide released. Related topics include the
Calvin–Benson cycle, carbon fixation, photosystems, cellular respiration, and artificial photosynthesis.
</item 1>

<item 2>
Photosynthesis
Light - dependent reactions Light - dependent reactions of photosynthesis at the thylakoid membrane Main article : Light - dependent reactions In the light -
dependent reactions ( mainly three vital reaction steps : ( I ) Hill and Bendall 's Z - scheme of cyclic and non-cyclic reactions , ( II ) Mitchell 's chemi -
osmosis for ATP synthesis and ( III ) Dolai 's S - state diagrams in water - splitting ) , one molecule of the pigment chlorophyll absorbs one photon and loses
one electron . This electron is passed to a modified form of chlorophyll called pheophytin , which passes the electron to a quinone molecule , starting the flow
of electrons down an electron transport chain that leads to the ultimate reduction of NADP to NADPH . In addition , this creates a proton gradient ( energy
gradient ) across the chloroplast membrane , which is used by ATP synthase in the synthesis of ATP . The chlorophyll molecule ultimately regains the electron it
lost when a water molecule is split in a process called photolysis , which releases a dioxygen ( O ) molecule as a waste product .
</item 2>

<item 3>
Photosynthesis
but canceling n water molecules from each side gives the net equation : CO + H O + photons → ( CH O ) + O carbon dioxide + water + light energy → carbohydrate +
oxygen Other processes substitute other compounds ( such as arsenite ) for water in the electron - supply role ; for example some microbes use sunlight to
oxidize arsenite to arsenate : The equation for this reaction is : CO + ( AsO 3 − 3 ) + photons → ( AsO 3 − 4 ) + CO carbon dioxide + arsenite + light energy →
arsenate + carbon monoxide ( used to build other compounds in subsequent reactions ) Photosynthesis occurs in two stages . In the first stage , light -
dependent reactions or light reactions capture the energy of light and use it to make the energy - storage molecules ATP and NADPH . During the second stage ,
the light - independent reactions use these products to capture and reduce carbon dioxide . Most organisms that utilize oxygenic photosynthesis use visible
light for the light - dependent reactions , although at least three use shortwave infrared or , more specifically , far - red radiation . Some organisms employ
even more radical variants of photosynthesis .
</item 3>

<item 4>
Photosynthesis
Z scheme The `` Z scheme '' In plants , light - dependent reactions occur in the thylakoid membranes of the chloroplasts where they drive the synthesis of ATP
and NADPH . The light - dependent reactions are of two forms : cyclic and non-cyclic . In the non-cyclic reaction , the photons are captured in the light -
harvesting antenna complexes of photosystem II by chlorophyll and other accessory pigments ( see diagram at right ) . The absorption of a photon by the antenna
complex frees an electron by a process called photoinduced charge separation . The antenna system is at the core of the chlorophyll molecule of the photosystem
II reaction center . That freed electron is transferred to the primary electron - acceptor molecule , pheophytin . As the electrons are shuttled through an
electron transport chain ( the so - called Z - scheme shown in the diagram ) , it initially functions to generate a chemiosmotic potential by pumping proton
cations ( H ) across the membrane and into the thylakoid space .
</item 4>

<item 5>
Photosynthesis uses light to convert carbon dioxide into carbohydrates, unlike cellular respiration, which oxidizes carbohydrates to release energy; the
processes use different reactions and cellular compartments. In oxygenic photosynthesis, water donates electrons and oxygen is released (net: CO₂ + H₂O + light
→ carbohydrate + O₂). Light-dependent reactions produce ATP and NADPH; light-independent reactions use them to capture and reduce CO₂. Some microbes use
alternative electron donors, such as arsenite, and some
</item 5>

<item 6>
Photosynthesis
The overall equation for the light - independent reactions in green plants is 3 CO + 9 ATP + 6 NADPH + 6 H → C H O - phosphate + 9 ADP + 8 P + 6 NADP + 3 H O
Overview of the Calvin cycle and carbon fixation Carbon fixation produces the intermediate three - carbon sugar product , which is then converted to the final
carbohydrate products . The simple carbon sugars produced by photosynthesis are then used in the forming of other organic compounds , such as the building
material cellulose , the precursors for lipid and amino acid biosynthesis , or as a fuel in cellular respiration . The latter occurs not only in plants but also
in animals when the energy from plants is passed through a food chain . The fixation or reduction of carbon dioxide is a process in which carbon dioxide
combines with a five - carbon sugar , ribulose 1 , 5 - bisphosphate , to yield two molecules of a three - carbon compound , glycerate 3 - phosphate , also known
as 3 - phosphoglycerate .
</item 6>

<item 7>
Photosynthesis
The cyclic reaction takes place only at photosystem I. Once the electron is displaced from the photosystem , the electron is passed down the electron acceptor
molecules and returns to photosystem I , from where it was emitted , hence the name cyclic reaction . Water photolysis Main articles : Photodissociation and
Oxygen evolution The NADPH is the main reducing agent produced by chloroplasts , which then goes on to provide a source of energetic electrons in other cellular
reactions . Its production leaves chlorophyll in photosystem I with a deficit of electrons ( chlorophyll has been oxidized ) , which must be balanced by some
other reducing agent that will supply the missing electron . The excited electrons lost from chlorophyll from photosystem I are supplied from the electron
transport chain by plastocyanin . However , since photosystem II is the first step of the Z - scheme , an external source of electrons is required to reduce its
oxidized chlorophyll a molecules . The source of electrons in green - plant and cyanobacterial photosynthesis is water .
</item 7>

<item 8>
Photosynthesis converts light, CO₂ and water into carbohydrates, usually releasing oxygen. It supplies most biospheric energy and nearly all organic matter:
roughly 100–115 billion tonnes of carbon and 130 terawatts globally each year. In chloroplast or bacterial membranes, photosystems drive electron transport,
water splitting, ATP and NADPH production; the Calvin cycle uses these to fix carbon. Pigments capture light; C₄ and CAM pathways reduce photorespiration. Early
photosynthesis was likely
</item 8>

<item 9>
Photosynthesis is the process by which plants, algae, and cyanobacteria use light energy to convert carbon dioxide and water into energy-storing carbohydrates.
It transforms sunlight into chemical energy that organisms can later use. Water is split during the process, usually releasing oxygen as a by-product. These
organisms are called photoautotrophs because they make food from carbon dioxide and water using light.
</item 9>

<item 10>
Photosynthesis sustains atmospheric oxygen and provides nearly all organic compounds and energy for life. Light is absorbed by chlorophyll-containing reaction
centres in plant chloroplasts or bacterial membranes. Light-dependent reactions split water, releasing oxygen, and use the resulting hydrogen and electrons to
produce NADPH and ATP, which store energy for cells.
</item 10>
=== END ITEM id=d0feb1dc4412319e ===

=== ITEM id=56067dfaa9637186 ===
Gold sentences and facts:
Sentence 0: Duncan 's diagnosis was publicly confirmed during a CDC news conference the same day . That evening , Duncan reported feeling better and requested to watch a movie . The following morning , Duncan was breathing rapidly and complaining of `` pain all over '' . By the afternoon , however , he was able to eat , and the doctor noted that he was feeling better . The next day , October 3 , Duncan again reported feeling abdominal pain . That evening , the hospital contacted Chimerix , a biotechnology company developing Brincidofovir to combat the disease . The next day , Duncan 's organs were failing , and he was intubated to help him breathe . In the afternoon , the hospital began administering Brincidofovir . Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock . On October 7 , the hospital reported that Duncan 's condition was improving . However , Duncan died at 7 : 51 a.m. on October 8 , becoming the first person to die in the United States of Ebola virus disease and the index patient for the later infections of nurses Pham and Vinson .
  Fact 0.0: The CDC confirmed Duncan's Ebola diagnosis at a news conference on the same day.
  Fact 0.1: That evening, Duncan reported feeling better and requested to watch a movie.
  Fact 0.2: The following morning, Duncan was breathing rapidly and complaining of 'pain all over'.
  Fact 0.3: By the afternoon, Duncan was able to eat.
  Fact 0.4: By the afternoon, the doctor noted that Duncan was feeling better.
  Fact 0.5: On October 3, Duncan again reported feeling abdominal pain.
  Fact 0.6: That evening, October 3, the hospital contacted Chimerix.
  Fact 0.7: Chimerix is a biotechnology company developing Brincidofovir to combat the disease.
  Fact 0.8: On October 4, Duncan's organs were failing.
  Fact 0.9: On October 4, Duncan was intubated to help him breathe.
  Fact 0.10: On October 4 in the afternoon, the hospital began administering Brincidofovir to Duncan.
  Fact 0.11: Nurses Nina Pham and Amber Joy Vinson continued to care for Duncan around the clock.
  Fact 0.12: On October 7, the hospital reported that Duncan's condition was improving.
  Fact 0.13: Duncan died at 7:51 a.m. on October 8.
  Fact 0.14: Duncan became the first person to die in the United States of Ebola virus disease.
  Fact 0.15: Duncan was the index patient for the later infections of nurses Pham and Vinson.
Sentence 1: In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola '' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first was reported in September 2014 . Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered .
  Fact 1.0: In December 2014, Ebola virus cases occurred in the United States due to four laboratory-confirmed cases of Ebola virus disease in the United States.
  Fact 1.1: Ebola virus disease is commonly known as Ebola.
  Fact 1.2: Eleven Ebola cases have been reported.
  Fact 1.3: The eleven reported cases include the four cases diagnosed in the United States and seven cases medically evacuated from other countries.
  Fact 1.4: The first of the eleven cases was reported in September 2014.
  Fact 1.5: Nine of the eleven cases contracted the disease outside the US and traveled into the country.
  Fact 1.6: The nine people traveled into the country either as regular airline passengers or as medical evacuees.
  Fact 1.7: Two of those nine people died.
  Fact 1.8: Two people have contracted Ebola in the United States.
  Fact 1.9: Both people who contracted Ebola in the United States were nurses who treated an Ebola patient.
  Fact 1.10: Both nurses have recovered.
Sentence 2: On September 30 , 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 , Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were :
  Fact 2.0: The CDC announced on September 30, 2014 that Thomas Eric Duncan had been diagnosed with Ebola in Dallas, Texas.
  Fact 2.1: Thomas Eric Duncan was reportedly 42 years old.
  Fact 2.2: CDC reports later corrected Thomas Eric Duncan's age to 45 years old.
  Fact 2.3: Thomas Eric Duncan was a Liberian national.
  Fact 2.4: Thomas Eric Duncan was visiting the United States from Liberia.
  Fact 2.5: Thomas Eric Duncan had been visiting family in Dallas.
  Fact 2.6: Thomas Eric Duncan was treated at Texas Health Presbyterian Hospital Dallas.
  Fact 2.7: By October 4, Thomas Eric Duncan's condition had deteriorated from 'serious but stable' to 'critical'.
  Fact 2.8: Thomas Eric Duncan died of Ebola on October 8.
  Fact 2.9: Three other cases were diagnosed in the United States as of October 2014, and they are listed after this sentence.
Sentence 3: Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
  Fact 3.0: The page is about Ebola virus disease in the U.S.
  Fact 3.1: The page includes a map of Ebola cases and infrastructure throughout the U.S.
  Fact 3.2: The page has categories for cases contracted in the U.S., cases first diagnosed in the U.S., and cases evacuated to the U.S. from other countries.
  Fact 3.3: The table lists 7 cases evacuated to the U.S. from other countries.
  Fact 3.4: The table lists 11 total cases.
  Fact 3.5: The table has a 'Deaths' label.
  Fact 3.6: The table lists 9 recoveries from Ebola.
  Fact 3.7: The table lists 0 active cases.

Retrieved items:
<item 1>
Ebola virus cases in the United States
In addition , on September 21 , a CDC employee , who showed no symptoms of the disease , was flown back to the United States as a preventative measure after low
- risk exposure with a healthcare worker ; he posed no risks to his family or the United States , and he is not known to have died . On September 28 , a fourth
American doctor was admitted to National Institutes of Health hospital . On October 2 , NBC News photojournalist Ashoka Mukpo , covering the outbreak in Liberia
, tested positive for Ebola after showing symptoms . Four other members of the NBC team , including physician Nancy Snyderman , were being closely monitored for
symptoms . Mukpo was evacuated on October 6 to the University of Nebraska Medical Center for treatment in their isolation unit . On October 21 , Mukpo was
declared Ebola - free and allowed to return to his home in Rhode Island . A government official in Sierra Leone announced on November 13 , that a doctor from
Sierra Leone , a permanent resident of the United States married to a U.S. citizen , would be transported to the Nebraska Medical Center for treatment for Ebola
. The doctor , identified later as Dr.
</item 1>

<item 2>
Ebola virus cases in the United States
October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan . October 23 , 2014 , physician Craig Spencer , diagnosed in New York City ; he had just
returned from working with Doctors Without Borders in Guinea , a country in west Africa . Hundreds of people were tested or monitored for potential Ebola virus
infection , but the two nurses were the only confirmed cases of locally transmitted Ebola . Public health experts and the Obama administration opposed
instituting a travel ban on Ebola endemic areas , stating that it would be ineffective and would paradoxically worsen the situation . No one both contracted and
died of Ebola virus disease while in the United States . No new cases have been diagnosed in the United States since Craig Spencer was released from a New York
City hospital on November 11 , 2014 . Contents ( hide ) 1 Cases diagnosed in U.S. 1.1 First case : Thomas Eric Duncan 1.1. 1 Thomas Eric Duncan in Liberia 1.1.
2 Duncan 's illness in Dallas 1.1. 3 Contact tracing 1.1.
</item 2>

<item 3>
Ebola virus cases in the United States
Medical evacuations from West Africa ( edit ) Local transmissions -- no deaths : Texas Initial cases -- deaths : Texas Initial cases -- no deaths : New York
Medically evacuated cases -- no deaths : Georgia , Maryland Medically evacuated cases -- deaths : Nebraska By the time the Ebola virus epidemic in West Africa
was over in June 2016 , seven Americans had been evacuated to the U.S. for treatment after contracting Ebola virus while working with medical teams trying to
stop the epidemic . Kent Brantly , a physician and medical director in Liberia for the aid group Samaritan 's Purse , and co-worker Nancy Writebol were infected
in July 2014 , while working in Monrovia . Both were flown to the United States at the beginning of August via the CDC 's Aeromedical Biological Containment
System for further treatment in Atlanta 's Emory University Hospital . On August 21 , Brantly and Writebol recovered and were discharged . Brantly has since
donated blood to three others with Ebola ( Sacra , Mukpo and Pham ) .
</item 3>

<item 4>
Ebola virus cases in the United States
Nine of the people contracted the disease outside the US and traveled into the country , either as regular airline passengers or as medical evacuees ; of those
nine , two died . Two people have contracted Ebola in the United States . Both were nurses who treated an Ebola patient ; both have recovered . On September 30
, 2014 , the CDC announced that Thomas Eric Duncan , a reportedly 42 - year - old ( later corrected by CDC reports as a 45 - year - old ) Liberian national
visiting the United States from Liberia , had been diagnosed with Ebola in Dallas , Texas . Duncan , who had been visiting family in Dallas , was treated at
Texas Health Presbyterian Hospital Dallas . By October 4 , Duncan 's condition had deteriorated from `` serious but stable '' to `` critical '' . On October 8 ,
Duncan died of Ebola. The other three cases diagnosed in the United States as of October 2014 were : October 11 , 2014 , a nurse , Nina Pham , who had provided
care to Duncan at the hospital . October 14 , 2014 , Amber Joy Vinson , another nurse who treated Duncan .
</item 4>

<item 5>
Ebola virus cases in the United States
WYFF4 . Retrieved October 3 , 2014 . Jump up ^ `` First US Ebola case Thomas Duncan ' critical ' '' . BBC News . October 4 , 2014 . Retrieved October 4 , 2014 .
Jump up ^ `` Texas Ebola patient dies '' . CNN . October 8 , 2014 . Retrieved October 8 , 2014 . ^ Jump up to : Fernandez , Manny ( October 12 , 2014 ) . ``
Texas Health Worker Tests Positive for Ebola '' . New York Times . Retrieved October 12 , 2014 . Jump up ^ `` Ebola outbreak : Second Texas health worker '
tests positive ' '' . BBC News . October 15 , 2014 . Retrieved October 15 , 2014 . ^ Jump up to : `` New York Doctor Just Back From Africa Has Ebola '' . NBC
News . October 23 , 2014 . Retrieved October 23 , 2014 . Jump up ^ Dan Diamond ( October 20 , 2014 ) .
</item 5>

<item 6>
Ebola virus cases in the United States
CBS News . November 13 , 2014 . Jump up ^ Sieff , Kevin ( November 17 , 2014 ) . `` A doctor 's mistaken Ebola test : ' We were celebrating ... Then everything
fell apart ' '' . Washington Post . Jump up ^ `` Martin Salia , Surgeon With Ebola , Arrives in Nebraska From Sierra Leone '' . New York Times . November 15 ,
2014 . Retrieved November 16 , 2014 . Jump up ^ Goodnough , Abby ; Trenchard , Tommy ( November 17 , 2014 ) . `` Doctor Being Treated for Ebola in Omaha Dies ''
. The New York Times . Jump up ^ U.S.
</item 6>

<item 7>
Ebola virus cases in the United States
Jump up ^ `` First diagnosed case of Ebola in the U.S. '' CNN . September 30 , 2014 . Retrieved October 27 , 2014 . Jump up ^ `` New Details About Ebola In
Dallas Revealed In CDC Report '' . HuffingtonPost . November 14 , 2014 . Retrieved November 14 , 2014 . Jump up ^ CDC confirms first ever Ebola case in United
States , cdc.gov ; accessed October 9 , 2014 . Jump up ^ Fernandez , Manny ; Philipps , Dave ( October 9 , 2014 ) . `` Death of Thomas Eric Duncan in Dallas
Fuels Alarm Over Ebola '' . The New York Times . Retrieved October 9 , 2014 . ^ Jump up to : Fernandez , Manny ; Onishi , Norimitsu ( October 1 , 2014 ) . ``
U.S. Patient Aided Ebola Victim in Liberia '' . The New York Times . ^ Jump up to : `` Thomas Eric Duncan : From healthy to Ebola '' . WYFF4 . Retrieved October
3 , 2014 .
</item 7>

<item 8>
Ebola virus cases in the United States
See also ( edit ) Ebola virus epidemic in Guinea Ebola virus epidemic in Liberia Ebola virus epidemic in Nigeria Ebola virus epidemic in Sierra Leone Ebola
virus disease in Spain List of Ebola outbreaks West Nile virus in the United States Notes ( edit ) Jump up ^ Duncan contracted Ebola in Liberia and was
diagnosed in the U.S. Two of the victims had had direct contact with Duncan . One victim was diagnosed in New York City ; his case is unrelated to the Texas
cases . ^ Jump up to : This takes into account the seven Americans who were medically evacuated from West Africa with Ebola . References ( edit ) Jump up ^ ``
Who are the American Ebola patients ? '' , CNN , October 6 , 2014 Jump up ^ `` Nebraska Hospital : Surgeon With Ebola Has Died '' . ABC News . November 17 ,
2014 . Retrieved November 17 , 2014 . Jump up ^ `` 2014 Ebola Outbreak - Case Count '' . CDC . December 15 , 2014 . Retrieved December 16 , 2014 . Jump up ^ ``
First diagnosed case of Ebola in the U.S. '' CNN .
</item 8>

<item 9>
Ebola virus cases in the United States
The New York Times . Jump up ^ U.S. Clinician With Ebola Under Care in Maryland , The New York Times , SHERI FINK , MARCH 13 , 2015 Jump up ^ Doctors treating
patient with Ebola at NIH downgrade condition to critical , The Washington Post , Brady Dennis , March 16 , 2015 Jump up ^ Americans exposed to Ebola patient
return from Africa for monitoring , Elizabeth Cohen , CNN , March 16 , 2015 Jump up ^ Sifferlin , Alexandra ( 26 March 2015 ) . `` American Patient With Ebola
Has Condition Upgraded '' . Time Inc . Retrieved 27 March 2015 . Jump up ^ Patient admitted with Ebola virus disease discharged from NIH Clinical Center Jump up
^ du Lac , J. Freedom ( October 13 , 2014 ) . `` The U.S. military 's new enemy : Ebola . Operation United Assistance is now underway . '' Washington Post .
Jump up ^ `` Operation United Assistance Helps in Liberian Ebola Fight '' . Jump up ^ ABC News . `` US Army to Quarantine Troops Who Were Fighting Ebola '' .
ABC News .
</item 9>

<item 10>
Ebola virus cases in the United States
Ebola virus cases in the United States - wikipedia Ebola virus cases in the United States Jump to : navigation , search This article is about Ebola virus cases
in the United States starting in 2014 . For the 2013 - 2016 outbreak in Africa , see Ebola virus epidemic in West Africa . For the U.S. discovery of a different
species of the Ebolavirus genus in 1989 , see Reston virus . Ebola virus disease in the U.S. Map of Ebola cases and infrastructure throughout the U.S. Cases
contracted in the U.S. Cases first diagnosed in U.S. Cases evacuated to U.S. from other countries 7 Total cases 11 Deaths Recoveries from Ebola 9 Active cases 0
In December 2014 , Ebola virus cases in the United States occurred due to four laboratory - confirmed cases of Ebola virus disease ( commonly known as `` Ebola
'' ) in the United States . Eleven cases have been reported , including these four cases and seven cases medically evacuated from other countries ; the first
was reported in September 2014 .
</item 10>
=== END ITEM id=56067dfaa9637186 ===

=== ITEM id=a38588bc92bffb9c ===
Gold sentences and facts:
Sentence 0: The Third Five - year Plan , stressed agriculture and improvement in the production of wheat , but the brief Sino - Indian War of 1962 exposed weaknesses in the economy and shifted the focus towards the defence industry and the Indian Army . In 1965 -- 1966 , India fought a War with Pakistan . There was also a severe drought in 1965 . The war led to inflation and the priority was shifted to price stabilisation . The construction of dams continued . Many cement and fertilizer plants were also built . Punjab began producing an abundance of wheat .
  Fact 0.0: The Third Five-year Plan stressed agriculture.
  Fact 0.1: The Third Five-year Plan stressed improvement in the production of wheat.
  Fact 0.2: The brief Sino-Indian War of 1962 exposed weaknesses in the economy.
  Fact 0.3: The brief Sino-Indian War of 1962 shifted the focus towards the defence industry and the Indian Army.
  Fact 0.4: India fought a war with Pakistan in 1965-1966.
  Fact 0.5: There was a severe drought in 1965.
  Fact 0.6: The war led to inflation.
  Fact 0.7: The priority was shifted to price stabilisation.
  Fact 0.8: The construction of dams continued.
  Fact 0.9: Many cement plants were built.
  Fact 0.10: Many fertilizer plants were built.
  Fact 0.11: Punjab began producing an abundance of wheat.

Retrieved items:
<item 1>
Sino-Indian War
China offered diplomatic support to Pakistan in this war but did not offer military support . In January 1966 , China condemned the Tashkent Agreement between
India and Pakistan as a Soviet - US plot in the region . In the Indo - Pakistani War of 1971 , Pakistan expected China to provide military support , but it was
left alone as India successfully helped the rebels in East Pakistan to found the new nation - state of Bangladesh . Involvement of other nations ( edit ) During
the conflict , Nehru wrote two desperate letters to U.S. President John F. Kennedy , requesting 12 squadrons of fighter jets and a modern radar system . These
jets were seen as necessary to beef up Indian air strength so that air - to - air combat could be initiated safely from the Indian perspective ( bombing troops
was seen as unwise for fear of Chinese retaliatory action ) . Nehru also asked that these aircraft be manned by American pilots until Indian airmen were trained
to replace them . These requests were rejected by the Kennedy Administration ( which was involved in the Cuban Missile Crisis during most of the Sino - Indian
War ) . The U.S.
</item 1>

<item 2>
Five-Year Plans of India
Indian Economy : Problems of development and planning . pune : Wishwa Prakashan . p. 676 . Jump up ^ `` National Development Council approves 12th Five Year
Plan '' . Indian Express . 2012 - 12 - 27 . Retrieved 2013 - 07 - 10 . Jump up ^ ( 1 ) External links ( edit ) Official website of the Planning Commission of
India India topics History Overviews Timeline Years Astronomy Economics Linguistics Maritime Mathematics Metallurgy Military Postal Science and technology
Pre-colonial Stone Age Indus Valley Civilization Vedic period Mahajanapadas Mauryas Middle kingdoms Hoysala Chola Pala Kakatiya Delhi Sultanate Vijayanagara
Mughals Marathas European trade Colonial Princely East India Company Plassey 1857 rebellion British Raj Railways Economy Army Zamindari Bengali renaissance
Political reforms Princely states Partition of Bengal Independence movement 1943 famine World War II Partition Republic Integration Non-Aligned Movement Five -
Year Plans Sino - Indian War Indo - Pakistani wars Green Revolution White Revolution Naxal Insurgency Smiling Buddha Space programme The Emergency Indian Peace
Keeping Force ( IPKF ) Economic
</item 2>

<item 3>
Sino-Indian War
The CIA later revealed that at that time the Chinese had neither the fuel nor runways long enough for using their air force effectively in Tibet . Indians in
general became highly sceptical of China and its military . Many Indians view the war as a betrayal of India 's attempts at establishing a long - standing peace
with China and started to question the once popular `` Hindi - Chini bhai - bhai '' ( meaning `` Indians and Chinese are brothers '' ) . The war also put an end
to Nehru 's earlier hopes that India and China would form a strong Asian Axis to counteract the increasing influence of the Cold War bloc superpowers . The
unpreparedness of the army was blamed on Defence Minister Menon , who resigned his government post to allow for someone who might modernise India 's military
further . India 's policy of weaponisation via indigenous sources and self - sufficiency was thus cemented . Sensing a weakened army , Pakistan , a close ally
of China , began a policy of provocation against India by infiltrating Jammu and Kashmir and ultimately triggering the Second Kashmir War with India in 1965 and
Indo - Pakistani war of 1971 .
</item 3>

<item 4>
Five-Year Plans of India
During 1966 -- 67 there was again the problem of drought . Equal priority was given to agriculture , its allied activities , and industrial sector . The
government of India declared `` Devaluation of Rupee '' to increase the exports of the country . The main reasons for plan holidays were the war , lack of
resources , and increase in inflation after that plan holiday was created . Fourth Plan ( 1969 -- 1974 ) ( edit ) At this time Indira Gandhi was the Prime
Minister . The Indira Gandhi government nationalised 14 major Indian banks and the Green Revolution in India advanced agriculture . In addition , the situation
in East Pakistan ( now Bangladesh ) was becoming dire as the Indo - Pakistan War of 1971 and Bangladesh Liberation War took funds earmarked for industrial
development . India also performed the Smiling Buddha underground nuclear test ( Pokhran - 1 ) in Rajasthan on May 18 , 1974 , partially in response to the
United States deployment of the Seventh Fleet in the Bay of Bengal . The fleet had been deployed to warn India against attacking West Pakistan and extending the
war .
</item 4>

<item 5>
Sino-Indian War
Mohammed Ali , External Affairs Minister of Pakistan , declared that massive Western aid to India in the Sino - Indian dispute would be considered an unfriendly
act towards Pakistan . As a result , Pakistan made efforts to improve its relations with China . The following year , China and Pakistan peacefully settled
disputes on their shared border , and negotiated the China - Pakistan Border Treaty in 1963 , as well as trade , commercial , and barter treaties . On 2 March
1963 , Pakistan conceded its northern claim line in Pakistani - controlled Kashmir to China in favor of a more southerly boundary along the Karakoram Range .
The border treaty largely set the border along the MacCartney - Macdonald Line . India 's military failure against China would embolden Pakistan to initiate the
Second Kashmir War with India . However , it effectively ended in a stalemate as Calvin states that the Sino - Indian War had caused the previously passive
government to take a stand on actively modernising India 's military . China offered diplomatic support to Pakistan in this war but did not offer military
support .
</item 5>

<item 6>
Sino-Indian War
The U.S. nonetheless provided non-combat assistance to Indian forces and planned to send the carrier USS Kitty Hawk to the Bay of Bengal to support India in
case of an air war . Some reports suggest a contradictory response from the U.S. According to former Indian diplomat G. Parthasarathy , `` only after we got
nothing from the US did arms supplies from the Soviet Union to India commence . '' In 1962 , President of Pakistan Ayub Khan made clear to India that Indian
troops could safely be transferred from the Pakistan frontier to the Himalayas . Aftermath ( edit ) China ( edit ) According to the China 's official military
history , the war achieved China 's policy objectives of securing borders in its western sector , as China retained de facto control of the Aksai Chin . After
the war , India abandoned the Forward Policy , and the de facto borders stabilised along the Line of Actual Control . According to James Calvin of Marine Corps
Command and Staff College , even though China won a military victory it lost in terms of its international image .
</item 6>

<item 7>
Sino-Indian War
However , Indian patrols south of the McMahon line were expelled and suffered casualties in the Chinese attack . Zhou also told Nixon that Chairman Mao ordered
the troops to return to show good faith . The Indian government maintains that the Chinese military could not advance further south due to logistical problems
and the cut - off of resource supplies . While Western nations did not view Chinese actions favourably because of fear of the Chinese and competitiveness ,
Pakistan , which had had a turbulent relationship with India ever since the Indian partition , improved its relations with China after the war . Prior to the
war , Pakistan also shared a disputed boundary with China , and had proposed to India that the two countries adopt a common defence against `` northern ''
enemies ( i.e. China ) , which was rejected by India . However , China and Pakistan took steps to peacefully negotiate their shared boundaries , beginning on 13
October 1962 , and concluding in December of that year . Pakistan also expressed fear that the huge amounts of western military aid directed to India would
allow it to threaten Pakistan 's security in future conflicts .
</item 7>

<item 8>
Five-Year Plans of India
In 1965 -- 1966 , India fought a War with Pakistan . There was also a severe drought in 1965 . The war led to inflation and the priority was shifted to price
stabilisation . The construction of dams continued . Many cement and fertilizer plants were also built . Punjab began producing an abundance of wheat . Many
primary schools were started in rural areas . In an effort to bring democracy to the grass - root level , Panchayat elections were started and the states were
given more development responsibilities . State electricity boards and state secondary education boards were formed . States were made responsible for secondary
and higher education . State road transportation corporations were formed and local road building became a state responsibility . The target growth rate was 5.6
% , but the actual growth rate was 2.4 % . Due to miserable failure of the Third Plan the government was forced to declare `` plan holidays '' ( from 1966 -- 67
, 1967 -- 68 , and 1968 -- 69 ) . Three annual plans were drawn during this intervening period . During 1966 -- 67 there was again the problem of drought .
</item 8>

<item 9>
Five-Year Plans of India
The fleet had been deployed to warn India against attacking West Pakistan and extending the war . The target growth rate was 5.6 % , but the actual growth rate
was 3.3 % . Fifth Plan ( 1974 -- 1978 ) ( edit ) The Fifth Five - Year Plan laid stress on employment , poverty alleviation ( Garibi Hatao ) , and justice . The
plan also focused on self - reliance in agricultural production and defence . In 1978 the newly elected Morarji Desai government rejected the plan . The
Electricity Supply Act was amended in 1975 , which enabled the central government to enter into power generation and transmission . The Indian national highway
system was introduced and many roads were widened to accommodate the increasing traffic . Tourism also expanded . The twenty - point programme was launched in
1975 . It was followed from 1974 to 1979 . The Minimum Needs Programme ( MNP ) was introduced in the first year of the Fifth Five Year Plan ( 1974 -- 78 ) . The
objective of the programme is to provide certain basic minimum needs and thereby improve the living standards of the people .
</item 9>

<item 10>
Sino-Indian War
China 's first nuclear weapon test in October 1964 and its support of Pakistan in the 1965 India Pakistan War tended to confirm the American view of communist
world objectives , including Chinese influence over Pakistan . Lora Saalman opined in a study of Chinese military publications , that while the war led to much
blame , debates and ultimately acted as causation of military modernisation of India but the war is now treated as basic reportage of facts with relatively
diminished interest by Chinese analysts . India ( edit ) U.S. Ambassador to India John Kenneth Galbraith and Prime Minister Nehru conferring at the time of the
conflict . This photograph was taken by the United States Information Service ( USIS ) and sent to President John F. Kennedy with a letter from Galbraith dated
9 November 1962 . The aftermath of the war saw sweeping changes in the Indian military to prepare it for similar conflicts in the future , and placed pressure
on Indian prime minister Jawaharlal Nehru , who was seen as responsible for failing to anticipate the Chinese attack on India . Indians reacted with a surge in
patriotism and memorials were erected for many of the Indian troops who died in the war .
</item 10>
=== END ITEM id=a38588bc92bffb9c ===

=== ITEM id=adc7b38a836341ea ===
Gold sentences and facts:
Sentence 0: The park lies in the north - east of South Africa , in the eastern parts of Limpopo and Mpumalanga provinces . Phalaborwa , Limpopo is the only town in South Africa that borders the Kruger National Park . It is one of the largest national parks in the world , with an area of 19,485 square kilometres ( 7,523 sq mi ) . The park is approximately 360 kilometres ( 220 mi ) long , and has an average width of 65 kilometres ( 40 mi ) . At its widest point , the park is 90 kilometres ( 56 mi ) wide from east to west .
  Fact 0.0: The Kruger National Park lies in the north-east of South Africa.
  Fact 0.1: The Kruger National Park lies in the eastern parts of Limpopo and Mpumalanga provinces.
  Fact 0.2: Phalaborwa, Limpopo is the only town in South Africa that borders the Kruger National Park.
  Fact 0.3: The Kruger National Park is one of the largest national parks in the world.
  Fact 0.4: The Kruger National Park has an area of 19,485 square kilometres (7,523 sq mi).
  Fact 0.5: The Kruger National Park is approximately 360 kilometres (220 mi) long.
  Fact 0.6: The Kruger National Park has an average width of 65 kilometres (40 mi).
  Fact 0.7: At its widest point, the Kruger National Park is 90 kilometres (56 mi) wide from east to west.
Sentence 1: Kruger National Park is one of the largest game reserves in Africa . It covers an area of 19,485 square kilometres ( 7,523 sq mi ) in the provinces of Limpopo and Mpumalanga in northeastern South Africa , and extends 360 kilometres ( 220 mi ) from north to south and 65 kilometres ( 40 mi ) from east to west . The administrative headquarters are in Skukuza . Areas of the park were first protected by the government of the South African Republic in 1898 , and it became South Africa 's first national park in 1926 .
  Fact 1.0: Kruger National Park is one of the largest game reserves in Africa.
  Fact 1.1: Kruger National Park covers an area of 19,485 square kilometres (7,523 sq mi).
  Fact 1.2: Kruger National Park is in the provinces of Limpopo and Mpumalanga.
  Fact 1.3: Kruger National Park is in northeastern South Africa.
  Fact 1.4: Kruger National Park extends 360 kilometres (220 mi) from north to south.
  Fact 1.5: Kruger National Park extends 65 kilometres (40 mi) from east to west.
  Fact 1.6: The administrative headquarters of Kruger National Park are in Skukuza.
  Fact 1.7: Areas of Kruger National Park were first protected by the government of the South African Republic in 1898.
  Fact 1.8: Kruger National Park became South Africa's first national park in 1926.
Sentence 2: Kruger National reserve IUCN category II ( national park ) Burchell 's zebra in a Kruger landscape Location of the park ( red area ) in South Africa Location Limpopo and Mpumalanga provinces , South Africa Nearest city Nelspruit ( southern ) Phalaborwa ( central ) Coordinates 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Coordinates : 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Area 19,485 km ( 7,523 sq mi ) Established 31 May 1926 ( 1898 ) Visitors 1,336,981 ( in 2004 ) Governing body South African National Parks www.sanparks.org/parks/kruger/
  Fact 2.0: Kruger National Reserve is IUCN category II (national park).
  Fact 2.1: Kruger National Reserve is located in Limpopo and Mpumalanga provinces, South Africa.
  Fact 2.2: The nearest city to Kruger National Reserve is Nelspruit (southern).
  Fact 2.3: The nearest city to Kruger National Reserve is Phalaborwa (central).
  Fact 2.4: Kruger National Reserve has coordinates 24°0′41''S 31°29′7''E / 24.01139°S 31.48528°E.
  Fact 2.5: Kruger National Reserve has an area of 19,485 km (7,523 sq mi).
  Fact 2.6: The infobox lists Kruger National Reserve as established on 31 May 1926, with 1898 given in parentheses.
  Fact 2.7: Kruger National Reserve had 1,336,981 visitors in 2004.
  Fact 2.8: Kruger National Reserve is governed by South African National Parks.
  Fact 2.9: The website for Kruger National Reserve is www.sanparks.org/parks/kruger/.

Retrieved items:
<item 1>
Graph query row:
park: Kruger National Park
relationship: LOCATED_IN
geographic_connection: Mpumalanga
</item 1>

<item 2>
Kruger National Park
Kruger National Park - Wikipedia Kruger National Park Jump to : navigation , search Kruger National reserve IUCN category II ( national park ) Burchell 's zebra
in a Kruger landscape Location of the park ( red area ) in South Africa Location Limpopo and Mpumalanga provinces , South Africa Nearest city Nelspruit (
southern ) Phalaborwa ( central ) Coordinates 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Coordinates : 24 ° 0 ′
41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Area 19,485 km ( 7,523 sq mi ) Established 31 May 1926 ( 1898 ) Visitors
1,336,981 ( in 2004 ) Governing body South African National Parks www.sanparks.
</item 2>

<item 3>
Kruger National Park
That motion , introduced together with another Volksraad member by the name of R.K. Loveday , and accepted for discussion in September 1895 by a majority of one
vote , resulted in the proclamation by Paul Kruger , president of the Transvaal Republic ( South African Republic ) , on 26 March 1898 , of a `` Government
Wildlife Park . '' This park would later be known as the Sabi Game Reserve . The park was initially created to control hunting and protect the diminished number
of animals in the park . James Stevenson - Hamilton became the first warden of the reserve in 1902 . The reserve was located in the southern one - third of the
modern park . Shingwedzi Reserve , named after the Shingwedzi River and now in northern Kruger National Park , was proclaimed in 1903 . During the following
decades all the native tribes were removed from the reserve and during the 1960s the last were removed at Makuleke in the Pafuri triangle . In 1926 , Sabie Game
Reserve , the adjacent Shingwedzi Game Reserve , and farms were combined to create Kruger National Park .
</item 3>

<item 4>
Kruger National Park
3 Red bush - willow and mopane veld 4.1. 4 Shrub mopane veld 4.2 Birds 4.3 Mammals 4.4 Reptiles 4.5 Amphibians and fish 5 Anti-poaching measures 5.1 Poachers
5.2 Rhino horn 5.3 Ivory stocks 5.4 Other 6 Accommodation and facilities 6.1 Rest camps 6.2 Designated private lodges 6.3 Bushveld camps 6.4 Restaurants 7 Gates
to the Kruger Park 8 Wilderness trails 9 Gallery 10 See also 11 References 12 External links History ( edit ) Main article : History of the Kruger National Park
Pre-reserve ( -- 1898 ) ( edit ) The area that the park covers today was part of the last wild frontier in the eastern half of Transvaal before the Second Anglo
- Boer War . Paul Kruger , President of the Republic of South Africa at the time , proclaimed the area , which was inhabited by the Tsonga people , a sanctuary
for the protection of its wildlife .
</item 4>

<item 5>
Graph query row:
park: Kruger National Park
relationship: LOCATED_IN
geographic_connection: Limpopo
</item 5>

<item 6>
Graph query row:
park: Kruger National Park
relationship: LOCATED_IN
geographic_connection: South Africa
</item 6>

<item 7>
Kruger National Park
981 ( in 2004 ) Governing body South African National Parks www.sanparks.org/parks/kruger/ Kruger National Park is one of the largest game reserves in Africa .
It covers an area of 19,485 square kilometres ( 7,523 sq mi ) in the provinces of Limpopo and Mpumalanga in northeastern South Africa , and extends 360
kilometres ( 220 mi ) from north to south and 65 kilometres ( 40 mi ) from east to west . The administrative headquarters are in Skukuza . Areas of the park
were first protected by the government of the South African Republic in 1898 , and it became South Africa 's first national park in 1926 . To the west and south
of the Kruger National Park are the two South African provinces of Limpopo and Mpumalanga . In the north is Zimbabwe , and to the east is Mozambique . It is now
part of the Great Limpopo Transfrontier Park , a peace park that links Kruger National Park with the Gonarezhou National Park in Zimbabwe , and with the Limpopo
National Park in Mozambique .
</item 7>

<item 8>
Kruger National Park
The land was given back to the Makuleke people , however , they chose not to resettle on the land but to engage with the private sector to invest in tourism ,
thus resulting in the building of several game lodges . In the late 1990s , the fences between the Kruger Park and Klaserie Game Reserve , Olifants Game Reserve
and Balule Game Reserve were dropped and incorporated into the Greater Kruger Park with 40 000 hectares added to the Reserve . In 2002 , Kruger National Park ,
Gonarezhou National Park in Zimbabwe , and Limpopo National Park in Mozambique were incorporated into a peace park , the Great Limpopo Transfrontier Park .
Location and geography ( edit ) Geography of the Kruger National Park Olifants River Luvuvhu and Limpopo rivers at Crookes Corner in Pafuri triangle The park
lies in the north - east of South Africa , in the eastern parts of Limpopo and Mpumalanga provinces . Phalaborwa , Limpopo is the only town in South Africa that
borders the Kruger National Park .
</item 8>

<item 9>
Kruger National Park
It is one of the largest national parks in the world , with an area of 19,485 square kilometres ( 7,523 sq mi ) . The park is approximately 360 kilometres ( 220
mi ) long , and has an average width of 65 kilometres ( 40 mi ) . At its widest point , the park is 90 kilometres ( 56 mi ) wide from east to west . To the
north and south of the park two rivers , the Limpopo and the Crocodile respectively , act as its natural boundaries . To the east the Lebombo Mountains separate
it from Mozambique . Its western boundary runs parallel with this range , roughly 65 kilometres ( 40 mi ) distant . The park varies in altitude between 200
metres ( 660 ft ) in the east and 840 metres ( 2,760 ft ) in the south - west near Berg - en - Dal . The highest point in the park is here , a hill called
Khandzalive . Several rivers run through the park from west to east , including the Sabie , Olifants , Crocodile , Letaba , Luvuvhu and Limpopo Rivers .
</item 9>

<item 10>
Kruger National Park
and min . temperatures in ° F Precipitation totals in inches The climate of the Kruger National Park and Lowveld is subtropical . Summer days are humid and hot
. The rainy season is from September until May . The Kruger National Park website lists September and October as the driest periods , culminating in rains late
in October . The dry winter season is the ideal time to visit this region for various reasons . There is less chance of contracting malaria and the days are
milder . Viewing wildlife is more rewarding as the vegetation is more sparse and animals are drawn to the waterholes to drink every morning and evening . Flora
and Fauna ( edit ) Vegetation ( edit ) Plants life in the park consists of four main areas : Natural landscape for much of the Park Thorn trees and red bush -
willow veld ( edit ) This area lies between the western boundary and roughly the centre of the park south of the Olifants River .
</item 10>
=== END ITEM id=adc7b38a836341ea ===

=== ITEM id=ebd57e4e70007de4 ===
Gold sentences and facts:
Sentence 0: Place Actor / Actress Episodes Finals Total score ( 175 ) 3 - 4 5 - 6 7 - 8 9 - 10 11 - 12 13 - 14 15 - 16 17 - 18 19 - 20 21 - 22 23 - 24 25 - 26 Aman Gandotra 8 12 14 14 7 + 3 = 10 9 + 3 = 12 10 + 3 = 13 10 + 4 = 14 10 + 5 = 15 10 + 5 = 15 10 + 4 = 14 Winner 141 Natasha Bharadwaj 9 18 14 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 Winner 166 2 / 3 Finalists Aashish Mehrotra 8 16 18 20 8 + 4 = 12 10 + 4 = 14 8 + 4 = 12 10 + 4 = 14 10 + 5 = 15 8 + 5 = 13 10 + 4 = 14 Runner up 156 Shruti Sharma 9 18 19 18 10 + 5 = 15 10 + 4 = 14 10 + 4 = 14 10 + 4 = 14 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 Runner up 166 Harshvardhan Deo NOT IN COMPETITION Debut 8 + 4 = 12 10 + 4 = 14 10 + 4 = 14 10 + 4 = 14 Runner up 54 Naina Singh 7 15 15 18 10 + 4 = 14 7 + 3 = 10 8 + 4 = 12 10 + 4 = 14 8 + 5 = 13 10 + 5 = 15 10 + 4 = 14 Runner up 147 Semifinalists Ansh Bagri 10 19 18 17 10 + 4 = 14 7 + 4 = 11 8 + 4 = 12 8 + 4 = 12 10 + 5 = 15 10 + 5 = 15 8 + 4 = 12 ELIMINATED 155 Lekha Prajapati 7 16 16 16 8 + 4 = 12 8 + 4 = 12 8 + 4 = 12 9 + 4 = 13 10 + 4 = 14 10 + 5 = 15 10 + 4 = 14 ELIMINATED 147 5 Shariq Nanda 9 16 14 14 10 + 3 = 13 8 + 4 = 12 8 + 3 = 11 10 + 4 = 14 8 + 5 = 13 ELIMINATED 116 Elisha Mayor NOT IN COMPETITION Debut 8 + 4 = 12 8 + 5 = 13 ELIMINATED 25 6 Tapan Singh NOT IN COMPETITION Debut 6 + 3 = 9 ELIMINATED 9 Simran Choudhary NOT IN COMPETITION Debut 8 + 4 = 12 ELIMINATED 12 7 Karan Taluja 7 17 16 16 6 + 3 = 9 8 + 4 = 12 6 + 3 = 9 ELIMINATED 86 Angela Krislinzki 9 18 20 16 8 + 5 = 13 6 + 3 = 9 6 + 3 = 9 ELIMINATED 97 8 Ahmed Masih 8 14 16 18 6 + 3 = 9 6 + 4 = 10 ELIMINATED 75 Kanikka Kapur 10 19 19 19 6 + 3 = 9 6 + 3 = 9 ELIMINATED 97 9 Harshvardhan Ahlawat 7 14 12 13 6 + 3 = 9 ELIMINATED 55 Pranati Rai Prakash 7 13 17 14 6 + 3 = 9 ELIMINATED 60 10 Jash Bahl 10 14 12 ELIMINATED 36 Heli Vyas 6 12 12 ELIMINATED 30 11 Shivank Chaudhary 6 12 ELIMINATED 18 Sahiba Bhasin 7 12 ELIMINATED 19 12 Karanraj Sharma 6 ELIMINATED 6 Laveena Keswani 6 ELIMINATED 6
  Fact 0.0: The table is headed Place, Actor / Actress, Episodes, Finals, and Total score (175), with round columns labeled 3-4 through 25-26.
  Fact 0.1: Aman Gandotra's total score is 141 and he is a Winner.
  Fact 0.2: Natasha Bharadwaj's total score is 166 and she is a Winner.
  Fact 0.3: Aashish Mehrotra is a Finalist with a Runner up placement and a total score of 156.
  Fact 0.4: Shruti Sharma is a Finalist with a Runner up placement and a total score of 166.
  Fact 0.5: Harshvardhan Deo was not in the competition and made a Debut, with a Runner up placement and a total score of 54.
  Fact 0.6: Naina Singh is a Finalist with a Runner up placement and a total score of 147.
  Fact 0.7: Ansh Bagri is a Semifinalist who was ELIMINATED with a total score of 155.
  Fact 0.8: Lekha Prajapati is a Semifinalist who was ELIMINATED with a total score of 147.
  Fact 0.9: Shariq Nanda was ELIMINATED with a total score of 116.
  Fact 0.10: Elisha Mayor was not in the competition, made a Debut, was ELIMINATED, and has a total score of 25.
  Fact 0.11: Tapan Singh was not in the competition, made a Debut, was ELIMINATED, and has a total score of 9.
  Fact 0.12: Simran Choudhary was not in the competition, made a Debut, was ELIMINATED, and has a total score of 12.
  Fact 0.13: Karan Taluja was ELIMINATED with a total score of 86.
  Fact 0.14: Angela Krislinzki was ELIMINATED with a total score of 97.
  Fact 0.15: Ahmed Masih was ELIMINATED with a total score of 75.
  Fact 0.16: Kanikka Kapur was ELIMINATED with a total score of 97.
  Fact 0.17: Harshvardhan Ahlawat was ELIMINATED with a total score of 55.
  Fact 0.18: Pranati Rai Prakash was ELIMINATED with a total score of 60.
  Fact 0.19: Jash Bahl was ELIMINATED with a total score of 36.
  Fact 0.20: Heli Vyas was ELIMINATED with a total score of 30.
  Fact 0.21: Shivank Chaudhary was ELIMINATED with a total score of 18.
  Fact 0.22: Sahiba Bhasin was ELIMINATED with a total score of 19.
  Fact 0.23: Karanraj Sharma was ELIMINATED with a total score of 6.
  Fact 0.24: Laveena Keswani was ELIMINATED with a total score of 6.
Sentence 1: India 's Next Superstars is a talent - searcher Indian reality show , which premiered on Star Plus and is streamed on Hotstar . Karan Johar and Rohit Shetty are the judges for the show . Aman Gandotra and Natasha Bharadwaj are declared winners of 2018 season , alongside Shruti Sharma who won a ' Special Mention ' award . Runners up in the male category were Aashish Mehrotra and Harshvardhan Deo and in the female category were Naina Singh and Shruti Sharma .
  Fact 1.0: India's Next Superstars is an Indian talent-searcher reality show.
  Fact 1.1: India's Next Superstars premiered on Star Plus.
  Fact 1.2: India's Next Superstars is streamed on Hotstar.
  Fact 1.3: Karan Johar and Rohit Shetty are the judges for India's Next Superstars.
  Fact 1.4: Aman Gandotra and Natasha Bharadwaj are declared winners of the 2018 season.
  Fact 1.5: Shruti Sharma is declared a winner of the 2018 season alongside Aman Gandotra and Natasha Bharadwaj.
  Fact 1.6: Shruti Sharma won a 'Special Mention' award.
  Fact 1.7: Aashish Mehrotra and Harshvardhan Deo were the runners up in the male category.
  Fact 1.8: Naina Singh and Shruti Sharma were the runners up in the female category.

Retrieved items:
<item 1>
India's Next Superstars
The fourth act of the evening had the strong message of inter-caste marriages , and how they are perceived in India . Delhi 's baby doll Kapur played the role
of a Muslim girl , and Vali that of a Hindu boy . Both of them were in love ; however , their families did not approve of it . The soulful performance moved the
audience and judges alike and Johar stated that it 's difficult to give comments immediately after such intense acts . Vali scored fourteen and Kapur scored a
high nineteen . The hosts played a weird game of ' ' yes / no ' ' with the judges and special guest Ranaut and Johar confessed that he has stolen toiletries
from hotel rooms . He also confessed that once he had stolen a pillow too . Ranaut was asked about her plans of marriage to which she stated that everyone will
get to know about it , as everything about her personal life comes up on the newspapers . Ranaut left the contestants with some motivating words , and stated
that she would like to meet them again soon . Choudhary and Bhasin were eliminated from the academy .
</item 1>

<item 2>
India's Next Superstars
Episode 25 This episode was aired on 07 Apr 2018 ( Saturday ) . The format of this episode was changed a bit . The contestants were scored on a forty pointer
scale ( ten marks for acting and ten marks for dancing by each judge ) and the scores were put in a secret ballot to be revealed at the end of the show . Shetty
declared that along with the movie contract with Dharma Productions , the contestants will win a cash prize of ten lakh rupees each and a gift hamper from close
up . Also the contestants will have to perform today along with an established actor in the industry . The trophy for the contest was presented to the audience
and Johar talked about the importance of the first award . He talked about the special bond he shares with his first filmfare award . There were audio visual
clippings shown in the background where the contestants talked about their first day in the academy , and how they are feeling now while leaving it . They
talked about the rehearsals , the friendship they shared , the fights they had and all other nostalgic moments .
</item 2>

<item 3>
India's Next Superstars
14 14 7 + 3 = 10 9 + 3 = 12 10 + 3 = 13 10 + 4 = 14 10 + 5 = 15 10 + 5 = 15 10 + 4 = 14 Winner 141 Natasha Bharadwaj 9 18 14 15 10 + 5 = 15 10 + 5 = 15 10 + 5 =
15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 10 + 5 = 15 Winner 166 2 / 3 Finalists Aashish Mehrotra 8 16 18 20 8 + 4 = 12 10 + 4 = 14 8 + 4 = 12 10 + 4 = 14 10 + 5 =
15 8 + 5 = 13 10 + 4 = 14 Runner up 156 Shruti Sharma 9 18 19 18 10 + 5 = 15 10 + 4 = 14 10 + 4 =
</item 3>

<item 4>
India's Next Superstars
Contents ( hide ) 1 Concept 2 Contestants 3 Guest appearance 4 Scoring chart 5 Synopsis of Episodes 6 Lights Camera Superstar cocktails 7 Superstars Ki
Paathshaala 8 References 9 External links Concept ( edit ) The new talent hunt show of Karan Johar and Rohit Shetty is concerned , ten males and ten females
will be living under one roof , and every day they will be assigned for tasks and will be judged on the basis of their performances . Additionally , two more
males and females , will join the competition as Wild - Cards . The two winners will be rewarded with a three - film deal with Dharma Productions house .
Contestants ( edit ) BOYS Names Notes Aashish Mehrotra RUNNER UP Ahmed Masih ELIMINATED ON 25 FEBRUARY 2018 Aman Gandotra WINNER Ansh Bagri SEMIFINALIST ;
</item 4>

<item 5>
India's Next Superstars
They asked quite a few questions to the renowned directors pointing out the flaws in their films . The contestants entered the show dancing on the popular song
' ' Swag se karenge sabka swagat ' ' from Tiger Zinda Hai along with the participant kids . In the first act of the evening , Gandotra played the role of Salim
Jahangir , and Bharadwaj played the role of Anarkali . The child actor Kartikey Raj played the role of Salim 's father Akbar . The comic timing of all the
actors made the act great and the dance ' ' chod chad ke apne Salim ki gali , anarkali disco chali ' ' added extra fun to the act . Shetty was very happy with
the performance and stated that Gandotra involved the rest of the contestants and audience in this act . Both the judges agreed that he was the heart of the act
. Gandotra scored a high nine and Bharadwaj scored perfect ten . The second act the evening was presented by Taluja and Prajapati and the child actor who
supported them was Twinkle Sharma .
</item 5>

<item 6>
India's Next Superstars
India 's next Superstars - Wikipedia India 's next Superstars Jump to : navigation , search This article may require copy editing for MOS issues (
Capitalization , bold , etc . ) . You can assist by editing it . ( April 2018 ) ( Learn how and when to remove this template message ) India 's Next Superstars
Genre Reality Based on Acting Presented by Rithvik Dhanjani Karan Wahi Judges INS Judges : Rohit Shetty Karan Johar INS Pathshala Judges : Mukesh Chhabra Mahesh
Bhatt Country of origin India No. of seasons No.
</item 6>

<item 7>
India's Next Superstars
Gandotra was impressed with the wigs and the earrings which were given to them along with the costumes to make their acts look perfect . References ( edit )
Jump up ^ `` Karan Johar , Rohit Shetty to judge a new talent show '' . DNA . Chaya Unnikrishnan . Retrieved November 3 , 2017 . Jump up ^ `` Masaba Gupta
designs Karan Johar 's outfits for India 's Next Superstars '' . Firstpost . IANS . Retrieved November 16 , 2017 . Jump up ^ `` Exclusive Aman Gandotra ,
Natasha Bharadwaj and Shruti Sharma were declared winners of India 's Next Superstars '' . The Indian Express . 2018 - 04 - 07 . Retrieved 2018 - 04 - 08 .
</item 7>

<item 8>
India's Next Superstars
Ace director and producer Karan Johar along with Rohit Shetty who is known for directing blockbuster movies started the show with a bang . The hosts of the show
were two popular faces of the entertainment industry Karan Wahi and Rithvik Dhanjani . Priyanka enters the stage citing a short poem ( Shayari ) Urdu Poetry --
' ' Jo desh ka guroor ban jaye , andhere mein noor ban jaye , jiski kasmein khakar mar mite janta , wohi hota hai ek asli superstar . ' ' It can be translated
as - ' ' Anyone who aspires to be a superstar becomes a pride of the nation and a shining light in the darkness . S / he reaches a stage of respect when the
common man starts worshipping him / her . ' ' And then she performed a sizzling number ' ' Desi Girl ' ' from her super hit movie Dostana . She lovingly tells
that Karan Johar 's Dharma ( religion ) and Karma ( work ) , both are films , and Rohit Shetty 's name has ' ' hit ' ' hidden in it .
</item 8>

<item 9>
India's Next Superstars
of seasons No. of episodes INS : 25 ( as of 7 April 2018 ) INS Pathshala : 60 ( as of 6 April 2018 ) Production Running time 1 hr 30 mins Production company ( s
) Dharma Productions Endemol Shine India Distributor Star India Release Original network STAR Plus Picture format 720p SDTV 1080p HDTV Original release 13
January ( 2018 - 01 - 13 ) -- 8 April 2018 External links Hotstar India 's Next Superstars is a talent - searcher Indian reality show , which premiered on Star
Plus and is streamed on Hotstar . Karan Johar and Rohit Shetty are the judges for the show . Aman Gandotra and Natasha Bharadwaj are declared winners of 2018
season , alongside Shruti Sharma who won a ' Special Mention ' award . Runners up in the male category were Aashish Mehrotra and Harshvardhan Deo and in the
female category were Naina Singh and Shruti Sharma .
</item 9>

<item 10>
India's Next Superstars
Quizzing in India Sell Me the Answer Thatt Antha Heli Singing Airtel Super Singer Airtel Super Singer Junior Fame Gurukul Idea Star Singer Indian Idol Sa Re Ga
Ma Pa ( Sa Re Ga Ma Pa L'il Champs ) Voice of India Lip Sing Battle Sports Box Cricket League BCL Punjab Zee 20 Cricket League Others Bigg Boss Bigg Boss Bangla
Bigg Boss Tamil Bigg Boss Kannada Bigg Boss Telugu Biggest Loser Jeetega India 's Got Talent Lead India Malayalee House National Bingo Night Sacch Ka Saamna
Defunct Ad Mad Show Antakshari -- The Great Challenge Baazi Kiski Biggest Loser Jeetega Boogie Woogie Champions of the World Dance Premier League Deal Ya No
Deal Fame X Idea Star Singer India 's Magic Star Iss Jungle Se Mujhe Bachao Jo Jeeta Wohi Super Star Junoon Lead India Mastermind India Perfect Bride National
Bingo Night Sawaal Dus Crore Ka Swayamwar Titan Antakshari Zee Cinestars Zor Ka Jhatka Retrieved from `` https://en.wikipedia.org/w/index.
</item 10>
=== END ITEM id=ebd57e4e70007de4 ===
