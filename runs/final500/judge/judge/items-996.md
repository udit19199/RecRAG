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

=== ITEM id=17a6e7a863abc377 ===
Gold sentences and facts:
Sentence 0: The formation of ATP from ADP and P is energetically unfavorable and would normally proceed in the reverse direction . In order to drive this reaction forward , ATPase couples ATP synthesis during cellular respiration to an electrochemical gradient created by the difference in proton ( H ) concentration across the mitochondrial membrane in eukaryotes or the plasma membrane in bacteria . During photosynthesis in plants , ATP is synthesized by ATPase using a proton gradient created in the thylakoid lumen through the thylakoid membrane and into the chloroplast stroma .
  Fact 0.0: The formation of ATP from ADP and P is energetically unfavorable.
  Fact 0.1: The formation of ATP from ADP and P would normally proceed in the reverse direction.
  Fact 0.2: ATPase couples ATP synthesis during cellular respiration to an electrochemical gradient in order to drive the reaction forward.
  Fact 0.3: The electrochemical gradient is created by the difference in proton (H) concentration across the mitochondrial membrane in eukaryotes.
  Fact 0.4: The electrochemical gradient is created by the difference in proton (H) concentration across the plasma membrane in bacteria.
  Fact 0.5: During photosynthesis in plants, ATP is synthesized by ATPase using a proton gradient.
  Fact 0.6: During photosynthesis in plants, the proton gradient is created in the thylakoid lumen through the thylakoid membrane.
  Fact 0.7: During photosynthesis in plants, the proton gradient passes into the chloroplast stroma.
Sentence 1: Located within the thylakoid membrane and the inner mitochondrial membrane , ATP synthase consists of two regions F and F. F causes rotation of F and is made of c - ring and subunits a , b , d , F6. F is made of α , β , γ , δ ( \ displaystyle \ alpha , \ beta , \ gamma , \ delta ) subunits . F has a water soluble par that can hydrolyze ATP . F on the other hand has mainly hydrophobic regions . F F creates a pathway for protons movement across the membrane .
  Fact 1.0: ATP synthase is located within the thylakoid membrane and the inner mitochondrial membrane.
  Fact 1.1: ATP synthase consists of two regions, F and F.
  Fact 1.2: The F region causes rotation of the F region.
  Fact 1.3: The F region is made of a c-ring and subunits a, b, d, and F6.
  Fact 1.4: The F region is made of alpha, beta, gamma, and delta subunits.
  Fact 1.5: The F region has a water soluble part that can hydrolyze ATP.
  Fact 1.6: The F region has mainly hydrophobic regions.
  Fact 1.7: The F and F regions together create a pathway for protons to move across the membrane.
Sentence 2: The crystal structure of the F showed alternating alpha and beta subunits ( 3 of each ) , arranged like segments of an orange around a rotating asymmetrical gamma subunit . According to the current model of ATP synthesis ( known as the alternating catalytic model ) , the transmembrane potential created by ( H+ ) proton cations supplied by the electron transport chain , drives the ( H+ ) proton cations from the intermembrane space through the membrane via the F region of ATP synthase . A portion of the F ( the ring of c - subunits ) rotates as the protons pass through the membrane . The c - ring is tightly attached to the asymmetric central stalk ( consisting primarily of the gamma subunit ) , causing it to rotate within the alpha beta of F causing the 3 catalytic nucleotide binding sites to go through a series of conformational changes that leads to ATP synthesis . The major F subunits are prevented from rotating in sympathy with the central stalk rotor by a peripheral stalk that joins the alpha beta to the non-rotating portion of F. The structure of the intact ATP synthase is currently known at low - resolution from electron cryo - microscopy ( cryo - EM ) studies of the complex . The cryo - EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F. Under the right conditions , the enzyme reaction can also be carried out in reverse , with ATP hydrolysis driving proton pumping across the membrane .
  Fact 2.0: The crystal structure of the F region showed alternating alpha and beta subunits, three of each.
  Fact 2.1: The alpha and beta subunits are arranged like segments of an orange around a rotating asymmetrical gamma subunit.
  Fact 2.2: The current model of ATP synthesis is known as the alternating catalytic model.
  Fact 2.3: In the alternating catalytic model, the transmembrane potential created by proton cations supplied by the electron transport chain drives proton cations from the intermembrane space through the membrane via the F region of ATP synthase.
  Fact 2.4: A portion of the F region, the ring of c-subunits, rotates as protons pass through the membrane.
  Fact 2.5: The c-ring is tightly attached to the asymmetric central stalk, which consists primarily of the gamma subunit.
  Fact 2.6: The c-ring causes the central stalk to rotate within the alpha and beta subunits of F.
  Fact 2.7: The rotation causes the 3 catalytic nucleotide binding sites to go through a series of conformational changes that leads to ATP synthesis.
  Fact 2.8: The major F subunits are prevented from rotating in sympathy with the central stalk rotor by a peripheral stalk.
  Fact 2.9: The peripheral stalk joins the alpha and beta subunits to the non-rotating portion of F.
  Fact 2.10: The structure of the intact ATP synthase is currently known at low resolution from electron cryo-microscopy (cryo-EM) studies of the complex.
  Fact 2.11: The cryo-EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F.
  Fact 2.12: Under the right conditions, the enzyme reaction can also be carried out in reverse.
  Fact 2.13: In reverse, ATP hydrolysis drives proton pumping across the membrane.

Retrieved items:
<item 1>
Cellular respiration
Aerobic respiration summed up 1 . Glycolysis : -- - 2 ATPs + Glucose → 2 Pyruvic Acid + 4 Hydrogen + 4 ATPs 2 . Formation of Acetyl CoA : -- - 2 Pyruvic Acid +
2 CoA → 2 Acetyl CoA + 2 Carbon Dioxide + 2 Hydrogen 3 . Krebs Cycle : -- - 2 Acetyl CoA + 3 O2 → 6 Hydrogen + 4 Carbon Dioxide + 2 ATPs 4 . Electron Transport
System : -- - 12 Hydrogen + 3 O2 → 6 Water + 32 ATPs Overall Reaction : -- - Glucose + 6 O2 → 6 Carbon Dioxide + 6 Water + 36 ATPs Glycolysis Out of the
cytoplasm it goes into the Krebs cycle with the acetyl CoA . It then mixes with CO and makes 2 ATP , NADH , and FADH . From there the NADH and FADH go into the
NADH reductase , which produces the enzyme .
</item 1>

<item 2>
Cellular respiration
The electrons are finally transferred to exogenous oxygen and , with the addition of two protons , water is formed . Efficiency of ATP production The table
below describes the reactions involved when one glucose molecule is fully oxidized into carbon dioxide . It is assumed that all the reduced coenzymes are
oxidized by the electron transport chain and used for oxidative phosphorylation . Step coenzyme yield ATP yield Source of ATP Glycolysis preparatory phase − 2
Phosphorylation of glucose and fructose 6 - phosphate uses two ATP from the cytoplasm .
</item 2>

<item 3>
ATP synthase
According to the current model of ATP synthesis ( known as the alternating catalytic model ) , the transmembrane potential created by ( H+ ) proton cations
supplied by the electron transport chain , drives the ( H+ ) proton cations from the intermembrane space through the membrane via the F region of ATP synthase .
A portion of the F ( the ring of c - subunits ) rotates as the protons pass through the membrane . The c - ring is tightly attached to the asymmetric central
stalk ( consisting primarily of the gamma subunit ) , causing it to rotate within the alpha beta of F causing the 3 catalytic nucleotide binding sites to go
through a series of conformational changes that leads to ATP synthesis . The major F subunits are prevented from rotating in sympathy with the central stalk
rotor by a peripheral stalk that joins the alpha beta to the non-rotating portion of F. The structure of the intact ATP synthase is currently known at low -
resolution from electron cryo - microscopy ( cryo - EM ) studies of the complex .
</item 3>

<item 4>
Cellular respiration
The NADH pulls the enzyme 's electrons to send through the electron transport chain . The electron transport chain pulls H ions through the chain . From the
electron transport chain , the released hydrogen ions make ADP for an end result of 32 ATP . O attracts itself to the left over electron to make water . Lastly
, ATP leaves through the ATP channel and out of the mitochondria . Main article : Glycolysis Glycolysis is a metabolic pathway that takes place in the cytosol
of cells in all living organisms . This pathway can function with or without the presence of oxygen . In humans , aerobic conditions produce pyruvate and
anaerobic conditions produce lactate . In aerobic conditions , the process converts one molecule of glucose into two molecules of pyruvate ( pyruvic acid ) ,
generating energy in the form of two net molecules of ATP . Four molecules of ATP per glucose are actually produced , however , two are consumed as part of the
preparatory phase . The initial phosphorylation of glucose is required to increase the reactivity ( decrease its stability ) in order for the molecule to be
cleaved into two pyruvate molecules by the enzyme aldolase .
</item 4>

<item 5>
ATP synthase
Depiction of ATP synthase using the chemiosmotic proton gradient to power ATP synthesis through oxidative phosphorylation . In the 1960s through the 1970s ,
Paul Boyer , a UCLA Professor , developed the binding change , or flip - flop , mechanism theory , which postulated that ATP synthesis is dependent on a
conformational change in ATP synthase generated by rotation of the gamma subunit . The research group of John E. Walker , then at the MRC Laboratory of
Molecular Biology in Cambridge , crystallized the F catalytic - domain of ATP synthase . The structure , at the time the largest asymmetric protein structure
known , indicated that Boyer 's rotary - catalysis model was , in essence , correct . For elucidating this , Boyer and Walker shared half of the 1997 Nobel
Prize in Chemistry . The crystal structure of the F showed alternating alpha and beta subunits ( 3 of each ) , arranged like segments of an orange around a
rotating asymmetrical gamma subunit .
</item 5>

<item 6>
ATP synthase
Large - enough quantities of ATP cause it to create a transmembrane proton gradient , this is used by fermenting bacteria that do not have an electron transport
chain , but rather hydrolyze ATP to make a proton gradient , which they use to drive flagella and the transport of nutrients into the cell . In respiring
bacteria under physiological conditions , ATP synthase , in general , runs in the opposite direction , creating ATP while using the proton motive force created
by the electron transport chain as a source of energy . The overall process of creating energy in this fashion is termed oxidative phosphorylation . The same
process takes place in the mitochondria , where ATP synthase is located in the inner mitochondrial membrane and the F - part projects into mitochondrial matrix
. The consumption of ATP by ATP - synthase pumps proton cations into the matrix . Evolution ( edit ) The evolution of ATP synthase is thought to have been
modular whereby two functionally independent subunits became associated and gained new functionality . This association appears to have occurred early in
evolutionary history , because essentially the same structure and activity of ATP synthase enzymes are present in all kingdoms of life .
</item 6>

<item 7>
ATP synthase
ATP synthase - wikipedia ATP synthase Jump to : navigation , search ATP Synthase Molecular model of ATP synthase determined by X-ray crystallography . Stator is
not shown here . Identifiers EC number 3.6. 3.14 CAS number 9000 - 83 - 3 Databases IntEnz IntEnz view BRENDA BRENDA entry ExPASy NiceZyme view KEGG KEGG entry
MetaCyc metabolic pathway PRIAM profile PDB structures RCSB PDB PDBe PDBsum Gene Ontology AmiGO / QuickGO ( show ) Search PMC articles PubMed articles NCBI
proteins ATP synthase is an enzyme that creates the energy storage molecule adenosine triphosphate ( ATP ) . ATP is the most commonly used `` energy currency ''
of cells for most organisms . It is formed from adenosine diphosphate ( ADP ) and inorganic phosphate ( P ) .
</item 7>

<item 8>
ATP synthase
The overall reaction catalyzed by ATP synthase is : ADP + P + H ⇌ ATP + H O + H The formation of ATP from ADP and P is energetically unfavorable and would
normally proceed in the reverse direction . In order to drive this reaction forward , ATPase couples ATP synthesis during cellular respiration to an
electrochemical gradient created by the difference in proton ( H ) concentration across the mitochondrial membrane in eukaryotes or the plasma membrane in
bacteria . During photosynthesis in plants , ATP is synthesized by ATPase using a proton gradient created in the thylakoid lumen through the thylakoid membrane
and into the chloroplast stroma . ATP synthase consists of two main subunits , F and F , which has a rotational motor mechanism allowing for ATP production .
Because of its rotating subunit ATP Synthase is a molecular machine .
</item 8>

<item 9>
Cellular respiration
Isocitrate is modified to become α - ketoglutarate ( 5 carbons ) , succinyl - CoA , succinate , fumarate , malate , and , finally , oxaloacetate . The net gain
of high - energy compounds from one cycle is 3 NADH , 1 FADH , and 1 GTP ; the GTP may subsequently be used to produce ATP . Thus , the total yield from 1
glucose molecule ( 2 pyruvate molecules ) is 6 NADH , 2 FADH , and 2 ATP . Oxidative phosphorylation Main articles : Oxidative phosphorylation , Electron
transport chain , Electrochemical gradient , and ATP synthase In eukaryotes , oxidative phosphorylation occurs in the mitochondrial cristae . It comprises the
electron transport chain that establishes a proton gradient ( chemiosmotic potential ) across the boundary of inner membrane by oxidizing the NADH produced from
the Krebs cycle . ATP is synthesized by the ATP synthase enzyme when the chemiosmotic gradient is used to drive the phosphorylation of ADP .
</item 9>

<item 10>
ATP synthase
This may have evolved to carry out the reverse reaction and act as an ATP synthase . In different species ( edit ) E coli ( edit ) E. coli ATP synthase is the
simplest known form of ATP synthase , with 8 different subunit types . Yeast ( edit ) Yeast ATP synthase is one of the best - studied eukaryotic ATP synthases ;
and five F , eight F subunits , and seven associated proteins have been identified . Most of these proteins have homologues in other eukaryotes . Plant ( edit )
In plants , ATP synthase is also present in chloroplasts ( CF F - ATP synthase ) . The enzyme is integrated into thylakoid membrane ; the CF - part sticks into
stroma , where dark reactions of photosynthesis ( also called the light - independent reactions or the Calvin cycle ) and ATP synthesis take place . The overall
structure and the catalytic mechanism of the chloroplast ATP synthase are almost the same as those of the mitochondrial enzyme . However , in chloroplasts , the
proton motive force is generated not by respiratory electron transport chain but by primary photosynthetic proteins .
</item 10>
=== END ITEM id=17a6e7a863abc377 ===

=== ITEM id=4eb3fd04927497af ===
Gold sentences and facts:
Sentence 0: John Brown 's raid on Harper 's Ferry ( also known as John Brown 's raid or The raid on Harper 's Ferry ) was an effort by armed abolitionist John Brown to initiate an armed slave revolt in 1859 by taking over a United States arsenal at Harpers Ferry , Virginia . Brown 's party of 22 was defeated by a company of U.S. Marines , led by First Lieutenant Israel Greene . Colonel Robert E. Lee was in overall command of the operation to retake the arsenal . John Brown had originally asked Harriet Tubman and Frederick Douglass , both of whom he had met in his transformative years as an abolitionist in Springfield , Massachusetts , to join him in his raid , but Tubman was prevented by illness and Douglass declined , as he believed Brown 's plan would fail .
  Fact 0.0: John Brown's raid on Harper's Ferry is also known as John Brown's raid.
  Fact 0.1: John Brown's raid on Harper's Ferry is also known as The raid on Harper's Ferry.
  Fact 0.2: John Brown's raid on Harper's Ferry was an effort by armed abolitionist John Brown to initiate an armed slave revolt in 1859.
  Fact 0.3: The raid aimed to start the armed slave revolt by taking over a United States arsenal at Harpers Ferry, Virginia.
  Fact 0.4: Brown's party of 22 was defeated by a company of U.S. Marines.
  Fact 0.5: The company of U.S. Marines that defeated Brown's party was led by First Lieutenant Israel Greene.
  Fact 0.6: Colonel Robert E. Lee was in overall command of the operation to retake the arsenal.
  Fact 0.7: John Brown had originally asked Harriet Tubman to join him in the raid.
  Fact 0.8: John Brown had originally asked Frederick Douglass to join him in the raid.
  Fact 0.9: John Brown had met Harriet Tubman and Frederick Douglass during his transformative years as an abolitionist in Springfield, Massachusetts.
  Fact 0.10: Harriet Tubman was prevented from joining the raid by illness.
  Fact 0.11: Frederick Douglass declined to join the raid.
  Fact 0.12: Frederick Douglass believed Brown's plan would fail.
Sentence 1: By 3 : 30 that afternoon , President James Buchanan ordered a company of U.S. Marines ( the only government troops in the immediate area ) to march on Harpers Ferry under the command of Brevet Colonel Robert E. Lee , lieutenant colonel of the 2nd U.S. Cavalry Regiment. Lee had been on leave from his regiment , stationed in Texas , when he was hastily called to lead the detachment and had to command it while wearing his civilian clothes .
  Fact 1.0: By 3:30 that afternoon, President James Buchanan ordered a company of U.S. Marines to march on Harpers Ferry.
  Fact 1.1: The Marines were the only government troops in the immediate area.
  Fact 1.2: The Marines marched on Harpers Ferry under the command of Brevet Colonel Robert E. Lee.
  Fact 1.3: Robert E. Lee was lieutenant colonel of the 2nd U.S. Cavalry Regiment.
  Fact 1.4: Lee had been on leave from his regiment when he was hastily called to lead the detachment.
  Fact 1.5: Lee's regiment was stationed in Texas.
  Fact 1.6: Lee had to command the detachment while wearing his civilian clothes.

Retrieved items:
<item 1>
John Brown's raid on Harpers Ferry
During the day , four townspeople were killed , including the mayor . Realizing his escape was cut , Brown took nine of his captives and moved into the smaller
engine house , which would come to be known as John Brown 's Fort . The raiders blocked entry of the windows and doors and traded sporadic gunfire with
surrounding forces . At one point Brown sent out his son , Watson , and Aaron Dwight Stevens with a white flag , but Watson was mortally wounded and Stevens was
shot and captured . The raid was rapidly failing . One of Brown 's men , William H. Leeman , panicked and made an attempt to flee by swimming across the Potomac
River , but he was shot and fatally injured while doing so . During the intermittent shooting , Brown 's other son , Oliver , was also hit ; he died after a
brief period . At around 3 : 00 p.m. a militia company led by Captain E.G. Alburtis arrived by train from Martinsburg , Virginia . Most of the militia members
were Baltimore & Ohio Railroad employees . The militia forced the raiders inside the engine house .
</item 1>

<item 2>
John Brown's raid on Harpers Ferry
John Brown 's raid on Harpers Ferry - wikipedia John Brown 's raid on Harpers Ferry Jump to : navigation , search John Brown 's Raid on Harper 's Ferry Part of
pre-Civil War conflicts Harper 's Weekly illustration of U.S. Marines attacking John Brown 's `` Fort '' Date October 16 -- 18 , 1859 Location Harpers Ferry , (
West ) Virginia Result U.S. victory Belligerents United States Abolitionist Insurgents Commanders and leaders Robert E. Lee Israel Greene Jeb Stuart John Brown
Strength 88 U.S. Marines Unknown number of Virginia Militia and Maryland Militia 8 white men 12 free black men 1 freed slave 1 fugitive slave Casualties and
losses U.S.
</item 2>

<item 3>
John Brown's raid on Harpers Ferry
, Collinsville , CT ) , in preparation for the raid . The United States Armory was a huge complex of buildings that manufactured small arms for the U.S. Army (
1801 -- 1861 ) , with an Arsenal ( storehouse ) that was thought to contain 100,000 muskets and rifles at the time . An illustration of Brown trying to persuade
abolitionist Frederick Douglass to join him in the raid on Harpers Ferry . Douglass refused . Brown attempted to attract more black recruits . He tried
recruiting Frederick Douglass as a liaison officer to the slaves in a meeting held in a quarry at Chambersburg , Pa . It was at this meeting that ex-slave ``
Emperor '' Shields Green consented to join with John Brown on his attack on the United States Armory , Green stating to Douglass `` I believe I will go with the
old man '' . Douglass declined , indicating to Brown that he believed the raid was a suicide mission . The plan was `` an attack on the federal government ''
that `` would array the whole country against us ... You will never get out alive '' , he warned .
</item 3>

<item 4>
John Brown's raid on Harpers Ferry
A free black man was the first casualty of the raid . Hayward Shepherd , an African - American baggage handler on the train , confronted the raiders ; they shot
and killed him . For some reason , Brown let the train continue , and the conductor alerted the authorities down the line . Brown had been sure that he would
win the support of local slaves in joining the rebellion , but a massive uprising did not occur , because word had not been spread about the uprising , so the
slaves nearby did not know about it . Although the white townspeople soon began to fight back against the raiders , Brown 's men succeeded in capturing the
armory that evening . October 17 ( edit ) Army workers discovered Brown 's men early on the morning of October 17 . Local militia , farmers and shopkeepers
surrounded the armory . When a company of militia captured the bridge across the Potomac River , any route of escape for the raiders was cut off . During the
day , four townspeople were killed , including the mayor .
</item 4>

<item 5>
John Brown's raid on Harpers Ferry
Marines : 1 killed 1 wounded Virginia and Maryland Militia : Unknown 10 killed 7 captured 5 escaped Civilians : 6 killed 9 wounded John Brown 's raid on Harper
's Ferry ( also known as John Brown 's raid or The raid on Harper 's Ferry ) was an effort by armed abolitionist John Brown to initiate an armed slave revolt in
1859 by taking over a United States arsenal at Harpers Ferry , Virginia . Brown 's party of 22 was defeated by a company of U.S. Marines , led by First
Lieutenant Israel Greene . Colonel Robert E. Lee was in overall command of the operation to retake the arsenal . John Brown had originally asked Harriet Tubman
and Frederick Douglass , both of whom he had met in his transformative years as an abolitionist in Springfield , Massachusetts , to join him in his raid , but
Tubman was prevented by illness and Douglass declined , as he believed Brown 's plan would fail .
</item 5>

<item 6>
John Brown's raid on Harpers Ferry
October 19 ( edit ) Robert E. Lee made a synopsis of the events that took place at Harpers Ferry . According to Lee 's notes , Lee believed John Brown was a
madman , `` ... the plan ( raiding the Harpers Ferry Arsenal ) was the attempt of a fanatic or madman . '' Lee also believed that the blacks in the raid were
forced by Brown . `` The blacks , whom he ( John Brown ) forced from their homes in this neighborhood , as far as I could learn , gave him no voluntary
assistance . '' Lee attributed John Brown 's `` temporary success '' by creating panic and confusion and by `` magnifying '' the number of participants involved
in the raid . Aftermath ( edit ) See also : Virginia v. John Brown A contemporary newspaper illustration showing the interior of the engine house immediately
before the door is broken down by U.S. Marines John Brown wrote his last prophecy on December 2 of 1859 . Colonel Lee and John Stuart searched the surrounding
country for fugitives who had participated in the attack .
</item 6>

<item 7>
John Brown's raid on Harpers Ferry
Contents ( hide ) 1 Brown 's preparation 2 Advance knowledge of raid 3 The raid 3.1 October 16 3.2 October 17 3.3 October 18 3.4 October 19 4 Aftermath 5
Casualties 5.1 John Brown 's raiders 5.2 Others 6 Liberated slaves 7 See also 8 Notes 8.1 Citations 8.2 Bibliography 9 External links Brown 's preparation (
edit ) John Brown rented the Kennedy Farmhouse , with a small cabin nearby , 4 miles ( 6.4 km ) north of Harpers Ferry near the community of Dargan in
Washington County , Maryland , and took up residence under the name Isaac Smith . Brown came with a small group of men minimally trained for military action .
His group included 18 men besides himself ( 13 white men , 5 black men ) . Northern abolitionist groups sent 198 breech - loading . 52 caliber Sharps carbines (
`` Beecher 's Bibles '' ) and 950 pikes ( obtained in late September from Charles Blair of Collinsville Axe Co.
</item 7>

<item 8>
John Brown's raid on Harpers Ferry
2007 On Sunday night , October 16 , 1859 , Brown left four of his men behind as a rear - guard : his son , Owen Brown , Barclay Coppock , and Frank Meriam ; he
led the rest into the town of Harpers Ferry , Virginia . Brown detached a party under John Cook Jr. to capture Colonel Lewis Washington , great - grandnephew of
George Washington , at his nearby Beall - Air estate , some of his slaves , and two relics of George Washington : a sword allegedly presented to Washington by
Frederick the Great and two pistols given by Marquis de Lafayette , which Brown considered talismans . The party carried out its mission and returned via the
Allstadt House , where they took more hostages . Brown 's main party captured several watchmen and townspeople in Harpers Ferry . Brown 's men needed to capture
the weapons and escape before word could be sent to Washington . The raid was going well for Brown 's men . They cut the telegraph wire and seized a Baltimore &
Ohio train passing through . A free black man was the first casualty of the raid .
</item 8>

<item 9>
John Brown's raid on Harpers Ferry
The militia forced the raiders inside the engine house . They broke into the guardroom and freed over two dozen prisoners . Eight militiamen were wounded .
Alburtis said that he could have ended the raid with help from other citizens . By 3 : 30 that afternoon , President James Buchanan ordered a company of U.S.
Marines ( the only government troops in the immediate area ) to march on Harpers Ferry under the command of Brevet Colonel Robert E. Lee , lieutenant colonel of
the 2nd U.S. Cavalry Regiment. Lee had been on leave from his regiment , stationed in Texas , when he was hastily called to lead the detachment and had to
command it while wearing his civilian clothes . October 18 ( edit ) This section needs additional citations for verification . Please help improve this article
by adding citations to reliable sources . Unsourced material may be challenged and removed . ( October 2017 ) ( Learn how and when to remove this template
message ) Lee first offered the role of attacking the engine house to the local militia units on the spot . Both militia commanders declined , and Lee turned to
the Marines .
</item 9>

<item 10>
John Brown's raid on Harpers Ferry
Greene later recounted what events occurred next : Quicker than thought I brought my saber down with all my strength upon ( Brown 's ) head . He was moving as
the blow fell , and I suppose I did not strike him where I intended , for he received a deep saber cut in the back of the neck . He fell senseless on his side ,
then rolled over on his back . He had in his hand a short Sharpe 's cavalry carbine . I think he had just fired as I reached Colonel Washington , for the Marine
who followed me into the aperture made by the ladder received a bullet in the abdomen , from which he died in a few minutes . The shot might have been fired by
someone else in the insurgent party , but I think it was from Brown . Instinctively as Brown fell I gave him a saber thrust in the left breast . The sword I
carried was a light uniform weapon , and , either not having a point or striking something hard in Brown 's accouterments , did not penetrate . The blade bent
double . In three minutes , all of the raiders still alive were taken prisoner and the action was over .
</item 10>
=== END ITEM id=4eb3fd04927497af ===

=== ITEM id=fdffbfa973ad879b ===
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
About 20 deaths occur annually in the US due to AHTR . Epidemiology ( edit ) Acute hemolytic , immune mediated ( fatal ) - 1 per 250,000 - 600,000 Acute
hemolytic , immune mediated ( nonfatal ) - 1 per 6000 - 33,000 Acute hemolytic , nonimmune - Infrequent References ( edit ) Jump up ^ Covin RB , Evans KS ,
Olshock R , Thompson HW ( 2001 ) . `` Acute hemolytic transfusion reaction caused by anti-Coa '' . Immunohematology. 17 ( 2 ) : 45 -- 9 . PMID 15373591 . Jump
up ^ Hoffbrand , A.V. ; P.A.H. Moss ; J.E. Pettit ( 2006 ) . Essential Haematology : 5th Edition . Blackwell Publishing . ISBN 1 - 4051 - 3649 - 9 .
</item 1>

<item 2>
Acute hemolytic transfusion reaction
, lactated Ringer 's solution or normal saline ) , and osmotic diuresis with mannitol . If DIC is documented and bleeding requires treatment , transfusions of
frozen plasma , pooled cryoprecipitates for fibrinogen , and / or platelet concentrates may be indicated . In case of non-antibody mediated AHR This does not
require rigorous management . Diuresis induced by an infusion of normal saline until the intense red color of hemoglobinuria ceases is usually adequate
treatment . Prognosis ( edit ) The major complication is that the hemoglobin , released by the destruction of red blood cells , may cause acute kidney failure (
also known as the `` oliguric phase '' ) . About 20 deaths occur annually in the US due to AHTR .
</item 2>

<item 3>
Acute hemolytic transfusion reaction
The 2014 Annual SHOT Report ( 2015 ) ( PDF ) . SHOT . ISBN 978 - 0 - 9558648 - 7 - 2 . ^ Jump up to : Laura , Dean ( 2005 ) . Blood Groups and Red Cell Antigens
. Bethesda , United States : National Center for Biotechnology Information . Retrieved 4 October 2017 . Jump up ^ Harrison 's Principles of Internal Medicine (
18th ed . ) . p. 954 . Jump up ^ `` Transfusion Reactions '' . Retrieved 19 November 2012 . Jump up ^ `` Complications of Transfusion : Transfusion Medicine :
Merck Manual Professional '' .
</item 3>

<item 4>
Acute hemolytic transfusion reaction
Acute hemolytic transfusion reaction - wikipedia Acute hemolytic transfusion reaction Jump to : navigation , search This article may be too technical for most
readers to understand . Please help improve it to make it understandable to non-experts , without removing the technical details . ( January 2018 ) ( Learn how
and when to remove this template message ) An acute hemolytic transfusion reaction ( AHTR ) is a type of transfusion reaction that is associated with hemolysis
. It occurs very soon after the transfusion , and within 24 hrs post-transfusion . It can occur quickly upon transfusing a few milliliters , or up to 1 -- 2
hours post-transfusion . It is also known as an `` immediate hemolytic transfusion reaction '' . This is a medical emergency as it results from rapid
destruction of the donor red blood cells by host antibodies ( IgG , IgM ) . It is usually related to ABO blood group incompatibility - the most severe of which
often involves group A red cells being given to a patient with group O type blood .
</item 4>

<item 5>
Acute hemolytic transfusion reaction
Types ( edit ) Acute hemolytic transfusion reactions are divided into two types . Intravascular haemolysis ( edit ) The most common cause of this reaction is
ABO blood groups incompatibility , secondary to human errors . In this process , the donor 's red blood cells are destroyed by recipient 's preformed antibodies
through the activation of complement system . Thus , the donor 's red blood cells break open and releases haemoglobin into the blood . The haemoglobin is then
excreted into urine , causes haemoglobinuria . Meanwhile , haemoglobin can also pass through the liver system and process into bilirubin , which accumulates
under the skin , causing jaundice . The release of red blood cell debris and the activation of complement system causes uncontrolled clotting ( disseminated
intravascular coagulation ) and circulatory shock . Extravascular haemolysis ( edit ) In this case , the donor 's red blood cells are recognised and bound to
IgG antibodies .
</item 5>

<item 6>
Acute hemolytic transfusion reaction
It may also be caused by other non-ABO IgG antibodies such as Rh , Kell , or Duffy . The former results in a severe intravascular hemolysis and the latter
typically causes extravascular hemolysis . The reactions are mediated by cytokines like TNF , IL - 8 , monocyte chemoattractant protein , IL - 1 etc .
Non-immune mediated ( edit ) These occur when RBCs are damaged before transfusion , resulting in hemoglobinemia and hemoglobinuria . Clinical symptoms are nil
or milder . Diagnosis ( edit ) Differential diagnoses ( edit ) Disseminated intravascular coagulation Hemolytic anemia Pulmonary edema Septic shock Management (
edit ) In case of antibody - mediated acute hemolytic transfusion reactions , Immediately discontinue the transfusion while maintaining venous access for
emergency management . Anticipate low blood pressure , kidney failure , and DIC . Prophylactic measures to reduce the risk of renal failure may include low -
dose dopamine , vigorous hydration with intravenous crystalloid solutions ( e.g.
</item 6>

<item 7>
Acute hemolytic transfusion reaction
the wrong unit of blood being given to the patient ) . Contents ( hide ) 1 Epidemiology 2 Types 2.1 Intravascular haemolysis 2.2 Extravascular haemolysis 3
Signs and symptoms 4 Pathophysiology 4.1 Immune mediated 4.2 Non-immune mediated 5 Diagnosis 5.1 Differential diagnoses 6 Management 7 Prognosis 8 Epidemiology
9 References Epidemiology ( edit ) In the USA in 2011 there were 48 episodes of acute hemolysis due to a mismatch in ABO blood type ( 1 in 495,207 blood
components transfused ) and 168 episodes of hemolysis due to other causes ( 1 in 124,525 blood components transfused ) . In the UK in 2014 there were 18
episodes of hemolysis due to other causes ( 1 in 147,972 blood components transfused ) . Types ( edit ) Acute hemolytic transfusion reactions are divided into
two types .
</item 7>

<item 8>
Acute hemolytic transfusion reaction
ISBN 1 - 4051 - 3649 - 9 . Jump up ^ Molthan L , Matulewicz TJ , Bansal - Carver B , Benz EJ ( 1984 ) . `` An immediate hemolytic transfusion reaction due to
anti-C and a delayed hemolytic transfusion reaction due to anti-Ce + e : hemoglobinemia , hemoglobinuria and transient impaired renal function '' . Vox Sang. 47
( 5 ) : 348 -- 53 . doi : 10.1111 / j. 1423 - 0410.1984. tb04138. x . PMID 6438912 . Jump up ^ `` The 2011 National Blood Collection and Utilization Survey
Report '' ( PDF ) . Department of Health and Human Services . Retrieved 21 January 2016 . Jump up ^ Bolton - Maggs , PHB ; Poles , D ; et al . ( The Serious
Hazards of Transfusion ( SHOT ) Steering Group ) ( 2015 ) . The 2014 Annual SHOT Report ( 2015 ) ( PDF ) .
</item 8>

<item 9>
Acute hemolytic transfusion reaction
Properdin then binds to complement C3 in the donor blood , facilitating the reaction through the alternate pathway cascade . The donor cells also become coated
with IgG and are subsequently removed by macrophages in the reticuloendothelial system ( RES ) . Jaundice and disseminated intravascular coagulation ( DIC ) may
also occur . The most common cause is clerical error ( i.e. the wrong unit of blood being given to the patient ) .
</item 9>

<item 10>
Acute hemolytic transfusion reaction
Macrophages then recognise these IgG antibodies and engulf these red blood cells , removing the red blood cells from the blood circulation into liver and spleen
. These red blood cells are then destroyed by macrophages in these organs . Anti-Rh antibodies most commonly directed this type of reaction . This process is
much more controlled when compared with intravascular haemolysis . Liver has adequate capacity in processing the bilirubin . Therefore , jaundice rarely occurs
. Signs and symptoms ( edit ) Early signs are fever , low blood pressure , anxiety , and red - colored urine . Late signs are generalized bleeding , caused by
disseminated intravascular coagulation , and low blood pressure . Laboratory assessment is based on a positive Direct Antiglobulin Test ( DAT ) , a decrease in
serum haptoglobin , and an increase in blood levels of the enzyme lactate dehydrogenase and indirect bilirubin levels . Pathophysiology ( edit ) Immune mediated
( edit ) This may be caused by preformed IgM anti-A , anti-B or both .
</item 10>
=== END ITEM id=fdffbfa973ad879b ===

=== ITEM id=b6f874cc790e40b9 ===
Gold sentences and facts:
Sentence 0: In 2012 , Maryse was featured on the cover of Le Journal de Montréal , and on MDA Show of Strength with The Miz and other celebrities . In 2015 , Ouellet had roles in Sharknado 3 : Oh Hell No ! and in Santa 's Little Helper , the latter starring Ouellet 's husband , The Miz . In the same year , she appeared on the E ! reality television series WAGS . In 2016 , it was announced that Maryse would appear as a main cast member on the sixth season of Total Divas , which prompted her return to WWE . In June 2017 , Maryse along with Miz were cover of the DUB Magazine with a Jeep car . In 2018 , Maryse along with The Miz will star in a reality television series titled Miz & Mrs .
  Fact 0.0: In 2012, Maryse was featured on the cover of Le Journal de Montréal.
  Fact 0.1: In 2012, Maryse was featured on MDA Show of Strength with The Miz.
  Fact 0.2: In 2012, Maryse was featured on MDA Show of Strength with other celebrities.
  Fact 0.3: In 2015, Ouellet had a role in Sharknado 3: Oh Hell No!
  Fact 0.4: In 2015, Ouellet had a role in Santa's Little Helper.
  Fact 0.5: Santa's Little Helper starred Ouellet's husband, The Miz.
  Fact 0.6: In 2015, Ouellet appeared on the E! reality television series WAGS.
  Fact 0.7: In 2016, it was announced that Maryse would appear as a main cast member on the sixth season of Total Divas.
  Fact 0.8: The 2016 announcement prompted Maryse's return to WWE.
  Fact 0.9: In June 2017, Maryse and Miz were on the cover of DUB Magazine with a Jeep car.
  Fact 0.10: In 2018, Maryse and The Miz will star in a reality television series titled Miz & Mrs.
Sentence 1: In late - 2011 , she announced plans for a clothing and jewelry line called House of Maryse , and later began working as a realtor . In November 2016 , she began starring in the reality television series Total Divas on E ! as part of the main cast . She and her husband The Miz will also star in their own reality show titled Miz & Mrs. that will premiere in 2018 .
  Fact 1.0: In late 2011, Maryse announced plans for a clothing and jewelry line called House of Maryse.
  Fact 1.1: Maryse later began working as a realtor.
  Fact 1.2: In November 2016, Maryse began starring in the reality television series Total Divas on E! as part of the main cast.
  Fact 1.3: Maryse and her husband The Miz will also star in their own reality show titled Miz & Mrs.
  Fact 1.4: The reality show Miz & Mrs. will premiere in 2018.

Retrieved items:
<item 1>
Maryse Ouellet
In late - 2011 , she announced plans for a clothing and jewelry line called House of Maryse , and later began working as a realtor . In November 2016 , she
began starring in the reality television series Total Divas on E ! as part of the main cast . She and her husband The Miz will also star in their own reality
show titled Miz & Mrs. that will premiere in 2018 . Contents ( hide ) 1 Early life and career 2 Professional wrestling career 2.1 World Wrestling Entertainment
/ WWE 2.1. 1 Diva Search and developmental territories ( 2006 -- 2007 ) 2.1. 2 Early feuds ( 2006 -- 2008 ) 2.1. 3 Divas Champion ( 2008 -- 2010 ) 2.1. 4
Managing Ted DiBiase and departure ( 2010 -- 2011 ) 2.2 Independent circuit ( 2012 ) 2.3 Return to WWE 2.3.
</item 1>

<item 2>
Maryse Ouellet
In 2018 , Maryse along with The Miz will star in a reality television series titled Miz & Mrs . Maryse has appeared in four WWE video games . She made her in -
game debut at WWE Smackdown vs. Raw 2010 , later appearing in WWE Smackdown vs. Raw 2011 , WWE ' 12 , and returned in WWE 2K18 . Filmography ( edit ) Film (
edit ) Year Title Role Notes Octobre Pouilleuse Film debut 2015 Sharknado 3 : Oh Hell No !
</item 2>

<item 3>
Maryse Ouellet
Following a hiatus due to the shoot of a WWE film , Maryse returned to WWE television on the June 27 episode of Raw , where she helped Miz to retain his
championship against Kane after she faked an ankle injury . After being drafted to Smackdown as part of the 2016 WWE draft which took place on July 19 , Ouellet
made her first appearance for the brand on July 26 , where she co-hosted a Miz TV segment along her husband before being interrupted by Randy Orton , who
challenged Miz to a non-title match , in which he was subsequently defeated . At Backlash , Maryse successfully helped Miz to retain the Intercontinental
Championship by spraying something at Dolph Ziggler whilst Miz distracted the referee . The following week , SmackDown General Manager Daniel Bryan announced a
rematch for the championship due Maryse 's interference at Backlash . On September 20 , Miz retained the championship over Ziggler .
</item 3>

<item 4>
Maryse Ouellet
On June 4 , at Extreme Rules , Maryse was in Miz 's corner when he won the WWE Intercontinental Championship for the seventh time by defeating Ambrose . In
September , following her pregnancy announcement , Maryse stopped appearing on television . On January 22 , 2018 , in the Raw 25 Years special episode , Maryse
was honored as part of a segment involving women considered legends that contributed to the company 's success , including the Bella Twins , Kelly Kelly ,
Lilian Garcia , Torrie Wilson , Michelle McCool , Terri Runnels , Maria Kanellis and the Hall of Famers Jacqueline and Trish Stratus . Other media ( edit )
Maryse and Miz in July 2016 In April 2007 , she appeared alongside fellow WWE Divas Ashley , Torrie Wilson , Brooke Adams , Layla , and Kelly Kelly in Timbaland
's music video `` Throw It On Me '' featuring The Hives . Ouellet also appeared in the January 2009 issue of Muscle & Fitness , along with Eve Torres and
Michelle McCool . She and John Morrison were interviewed on Eurosport in the same year .
</item 4>

<item 5>
Maryse Ouellet
She began appearing regularly for FWE , where she commentated during women 's matches . Return to WWE ( edit ) Managing the Miz ( 2016 -- present ) ( edit )
Maryse celebrating along The Miz after winning the Intercontinental Championship at a Raw event in April 2016 On April 4 , 2016 episode of Raw , the episode
after WrestleMania 32 , Maryse returned during a WWE Intercontinental Championship match between the champion Zack Ryder and her husband , The Miz . During the
match , Maryse slapped Ryder 's father , who was in the front row , in order to distract Ryder and allowed Miz to capture the championship for the fifth time ,
thus becoming his manager in the process , where the two then became an on - screen power couple . She formally re-introduced herself three days later on the
April 7 episode of SmackDown before once more helping Miz to retain his championship over Ryder . She then began to cut various promos with Miz as his host
during the `` Miz TV '' segments , while calling themselves the `` It '' Couple .
</item 5>

<item 6>
Maryse Ouellet
She is currently inactive due to her maternity leave . After spending years modeling , including winning Miss Hawaiian Tropic Canada in 2003 , Ouellet was hired
by the professional wrestling promotion WWE in August 2006 , after she participated in the WWE Diva Search competition . She spent time in Ohio Valley Wrestling
and Florida Championship Wrestling , WWE 's developmental territories , before being assigned to the SmackDown brand in 2008 . In December 2008 , Maryse won her
first WWE Divas Championship and held it for seven months , which is the third longest reign in the championship 's history . In April 2009 , she was drafted to
the Raw brand , and won the WWE Divas Championship for the second time in February 2010 , making her the first wrestler to have held the championship on more
than one occasion . In mid-2010 , she became the co-host of NXT and the manager of Ted DiBiase , prior to her release from the company in October 2011 . In
April 2016 , she returned to the company and became the manager of her husband The Miz , managing him to three Intercontinental Championship reigns .
</item 6>

<item 7>
Maryse Ouellet
. TMZ . February 22 , 2014 . Retrieved March 21 , 2014 . Jump up ^ `` The Miz and Maryse are expecting their first child '' . Retrieved March 19 , 2018 . Jump
up ^ Waldman , Jon ( April 25 , 2009 ) . `` Smackdown : UK gets a strong episode of SD ! '' . Slam ! Sports . Canadian Online Explorer . Retrieved February 14 ,
2010 . Jump up ^ David , Stephens ( January 4 , 2010 ) . `` WWE Raw Results -- 1 / 4 / 10 '' . WrestleView . Retrieved April 9 , 2016 . Jump up ^ David ,
Stephens ( May 17 , 2010 ) . `` WWE Raw Results -- 5 / 17 / 10 '' . WrestleView . Retrieved April 9 , 2016 . Jump up ^ Tedesco , Mike ( December 26 , 2008 ) .
</item 7>

<item 8>
Graph query row:
show: Miz & Mrs.
kind: reality show
showDate: 2018
relationship: None
relatedEvent: None
premiereDate: None
</item 8>

<item 9>
Maryse Ouellet
Maryse Ouellet - wikipedia Maryse Ouellet Jump to : navigation , search Maryse Ouellet Ouellet in April 2016 Birth name Maryse Ouellet ( 1983 - 01 - 21 )
January 21 , 1983 ( age 35 ) Montréal , Quebec , Canada Residence Austin , Texas , United States Spouse ( s ) The Miz ( m . 2014 ) Children Professional
wrestling career Ring name ( s ) Maryse Billed height 5 ft 8 in ( 1.73 m ) Billed weight 116 lb ( 53 kg ) Billed from Montreal , Quebec , Canada Trained by Al
Snow Dave Finlay Steve Keirn Tom Prichard Ricky Steamboat Debut August 24 , 2006 Maryse Mizanin ( / məˈriːs / ; née Ouellet ; born January 21 , 1983 ) is a
French Canadian professional wrestler , professional wrestling manager , actress , businesswoman and glamour model currently signed to WWE under the ring name
Maryse , performing on the SmackDown brand as the manager of her real - life husband The Miz .
</item 9>

<item 10>
Maryse Ouellet
E ! Online . Retrieved December 23 , 2017 . Jump up ^ `` Maryse Ouellet profile on famousVcard '' . Jump up ^ `` Maryse talks WWE release , restraining order ,
Divas Legend she wants to work with '' . Pro Wrestling Torch . January 5 , 2012 . Retrieved April 12 , 2012 . Jump up ^ `` More Diva dreams come true '' . WWE .
August 24 , 2006 . Retrieved June 27 , 2016 . Jump up ^ Nesbitt , Cherylann . `` Mike `` The Miz '' Mizanin Knows His Fiance is hotter `` . MTV . Retrieved
August 31 , 2013 . Jump up ^ Donnelly , Matthew Scott . `` Mike ' The Miz ' Mizanin ties the knot with a fellow WWE Champ '' . MTV . Retrieved March 21 , 2014 .
Jump up ^ `` WWE Superstar The Miz -- My Wedding Was Awesome ! '' . TMZ . February 22 , 2014 .
</item 10>
=== END ITEM id=b6f874cc790e40b9 ===

=== ITEM id=bdebe6b175b36c69 ===
Gold sentences and facts:
Sentence 0: Year City State , province , dept. , etc . Country Notes 7500 BC Tlapacoya Ixtapaluca Mexico 4000 BC Puerto Hormiga Culture Cartagena Colombia 3710 BC Aspero Norte Chico Peru 2627 BC Caral Norte Chico Peru 1500 BC Kaminaljuyu Guatemala city Guatemala 800 BC Cuicuilco Mexico City Mexico 700 BC Ticul Yucatán Mexico 500 BC Cholula Puebla Mexico 300 BC Teotihuacan México Mexico In the Valley of Mexico 200 Mitla Oaxaca Mexico 200 Acanceh Yucatan Mexico 500 Cuenca Azuay Ecuador 524 Ejutla Oaxaca Mexico ( 1 ) 550 Izamal Yucatan Mexico 600 Cahokia Illinois United States 713 Tula Hidalgo Mexico 1001 L'Anse aux Meadows Newfoundland and Labrador Canada Oldest known European ( Norse ) settlement in the Americas . Possibly founded by Leif Erikson , as part of Vinland , in 1001 AD . 1000 Acoma Pueblo and Taos Pueblo New Mexico United States 1050 Motul Yucatán Mexico 1050 Pachuca Hidalgo Mexico 1100 Cusco Cusco Province Peru 1100 Oraibi Arizona United States ( Hopi Reservation ) ( 2 ) 1168 Ecatepec Estado de Mexico Mexico 1259 Chimaulhuacan Mexico State Mexico 1313 Xalapa Veracruz Mexico 1325 Tenochtitlan Mexico City Mexico Largest pre-Columbian city in the Americas , later called Mexico City . 1450 Zuni Pueblo New Mexico United States ( 3 ) 1470 Iximche Chimaltenango Guatemala 1492 La Navidad Nord - Est Haiti First European settlement in the New World during the Age of Discovery . 1493 La Isabela Puerto Plata Dominican Republic 1498 Santo Domingo Distrito Nacional Dominican Republic Capital of the Dominican Republic . Oldest continuously inhabited European established settlement in the Americas . Founded in 1498 , by Bartholomew Columbus . 1502 Santa Cruz del Seibo El Seibo Dominican Republic 1504 Azua de Compostela Azua Dominican Republic 1504 Santiago de los Caballeros Santiago Dominican Republic 1505 Cotuí Sánchez Ramírez Dominican Republic 1506 Salvaleón de Higüey La Altagracia Dominican Republic 1508 Bonao Monseñor Nouel Dominican Republic 1508 Caparra Puerto Rico United States First city founded by Europeans , although not continuously inhabited , in the United States . Abandoned in 1521 with the removal of the capital to San Juan . 1510 Santa Maria la Antigua del Darien Urabá Colombia First city founded by Europeans on the continent of South America . 1510 Nombre de Dios Colon Panama Oldest continuously inhabited European established settlement in Panama . 1511 Baracoa Guantánamo Cuba Oldest continuously inhabited European established settlement in Cuba , former capital of Cuba . 1513 Bayamo Granma Cuba 1514 Sancti Spiritus Santi Spiritus Cuba 1514 Trinidad Santi Spiritus Cuba 1515 Santiago Santiago After Baracoa and before Havana , Santiago was the second capital of Cuba . Cuba 1519 Havana Havana Cuba 1519 La Villa Rica de la Vera Cruz , ( Veracruz ) Veracruz Mexico Oldest continuously inhabited European established settlement in Mexico . 1519 Panama City Panamá Panama First European established city on the Pacific Coast of the Americas . Founded in 1519 , at the present day ruins of Panama Viejo , it was sacked by the Welsh pirate Henry Morgan in 1671 , and resettled to nearby Casco Viejo , in 1673 . 1520 Hato Mayor del Rey Hato Mayor Dominican Republic 1515 Cumaná Sucre Venezuela Oldest continuously inhabited European established settlement in Venezuela . 1521 San Juan Puerto Rico United States Oldest continuously inhabited European established settlement in Puerto Rico , and in the United States . Oldest continuously inhabited state or territorial capital in the United States . 1524 Granada Granada Nicaragua Oldest continuously inhabited European established settlement in Nicaragua . 1524 Tecpán Guatemala Chimaltenango Guatemala First capital of Guatemala and oldest continuously inhabited European established settlement in Guatemala . 1524 Trujillo Colón Honduras Oldest continuously inhabited European established settlement in Honduras . 1525 Santa Marta Magdalena Colombia Oldest continuously inhabited European established settlement in Colombia . 1525 Acapulco Guerrero Mexico ( 4 ) 1525 San Salvador San Salvador El Salvador Founded in 1525 , rebuilt and changed locations twice afterwards ( 1528 and 1545 ) and oldest continuously inhabited European established settlement in El Salvador . 1526 Acámbaro Guanajuato Mexico 1526 Chimaltenango Chimaltenango Guatemala 1528 San Cristóbal de las Casas Chiapas Mexico Originally called Villa Real de Chiapa . 1528 Camagüey Camagüey Province Cuba Founded in 1515 the northern coast of Camagüey as `` Puerto del Príncipe '' , but moved in 1528 to present location . 1529 Maracaibo Zulia Venezuela 1530 San Blas Nayarit Mexico By Nuño Beltrán ( 5 ) 1531 Puebla Puebla Mexico by Fr . Toribio de Benavente ( `` Motolinía '' ) on 16 April ( 6 ) 1532 Oaxaca Oaxaca Mexico 1532 Piura Piura Peru 1532 São Vicente São Paulo Brazil Oldest continuously inhabited European established settlement in Brazil , oldest Portuguese established settlement in the New World . 1532 Tepic Nayarit Mexico As capital of Nueva Galicia ( 7 ) 1533 Cananéia São Paulo Brazil Second Oldest City in Brazil 1533 Cartagena de Indias Bolívar Colombia 1534 Otavalo Imbabura Ecuador 1534 Quito Pichincha Ecuador Oldest continuously inhabited European established settlement in Ecuador . 1534 Spanish Town Saint Catherine Jamaica Founded by the Spanish as Santiago de la Vega or St. Jago de la Vega . Oldest continuously inhabited European established settlement in Jamaica . 1534 Trujillo La Libertad Peru Oldest continuously inhabited European established settlement in Peru . 1535 Lima Lima Peru 1535 Igarassu Pernambuco Brazil Site of first European settlement in Brazil , the feitoria of Igarassu , in 1516 . 1535 Olinda Pernambuco Brazil One of the best - preserved colonial cities in Brazil . 1535 Vila Velha Espírito Santo Brazil 1535 Paria Oruro Bolivia First Spanish settlement in Bolivia 1536 Santiago de Cali Valle del Cauca Colombia 1536 Popayán Cauca Colombia 1536 San Pedro Sula Cortés Honduras 1536 Tupiza Potosí Bolivia 1537 Asunción Paraguay Oldest continuously inhabited European established settlement in Paraguay . 1537 Recife Pernambuco Brazil Capital of the state of Pernambuco . It is the oldest city among Brazil 's current state capitals . 1538 Bogotá Cundinamarca Colombia 1538 Guayaquil Guayas Ecuador 1538 Iguape São Paulo Brazil 1538 Sucre Chuquisaca Bolivia Originally called Ciudad de la Plata de la Nueva Toledo and the oldest continuously inhabited European established settlement in Bolivia . 1539 Tunja Boyacá Colombia 1539 Vélez Santander Colombia 1539 Pasto Nariño Colombia 1540 Arequipa Arequipa Peru 1540 Ayacucho Ayacucho Peru 1540 Campeche Campeche Mexico 1540 Yuriria Guanajuato Mexico 1540 Mompox Bolivar Colombia 1540 Supía Caldas Colombia 1540 Campeche Campeche Mexico Founded on the site of Mayan village Kimpech . 1541 Huánuco Huánuco Peru 1541 Moquegua Moquegua Peru 1541 Santiago Santiago Chile Oldest continuously inhabited European established settlement in Chile . 1541 Santafé de Antioquia Antioquia Colombia 1542 Guadalajara Jalisco Mexico 1542 Genaro Codina Zacatecas Mexico 1542 San Miguel de Allende Guanajuato Mexico 1542 Mérida Yucatan Mexico Built on the site of Mayan city T'ho . 1542 San Luis de la Paz Guanajuato Mexico 1543 Antigua Guatemala Sacatepéquez Guatemala 1543 Cobán Alta Verapaz Guatemala 1543 Santa Cruz Verapaz Alta Verapaz Guatemala 1545 Potosí Potosí Bolivia 1545 San Juan de los Remedios Villa Clara Province Cuba Remedios has the most uncertain foundation date in Cuba , 1545 has been accepted because it is when the village erected its town hall , but its origin has been studied and documented back to 1513 . 1545 Riohacha La Guajira Colombia 1546 Santos São Paulo Brazil 1546 Neyba Bahoruco Dominican Republic 1546 Tepezalá Aguascalientes Mexico 1548 Asientos Aguascalientes Mexico 1548 La Paz La Paz Bolivia 1548 Pánuco Zacatecas Mexico 1548 Zacatecas Zacatecas Mexico 1549 Salvador Bahia Brazil 1550 Concepción Concepción Chile 1550 Valledupar Cesar Colombia 1550 Ibagué Tolima Colombia 1551 Vitória Espírito Santo Brazil 1551 Mocoa Putumayo 1552 Sonsonate Sonsonate El Salvador 1553 Santiago del Estero Santiago del Estero Argentina Oldest continuously inhabited European established settlement in Argentina . 1553 São Bernardo do Campo São Paulo Brazil 1554 Embu São Paulo Brazil 1554 São Paulo São Paulo Brazil 1555 Saín Alto Zacatecas Mexico 1556 Chalchihuites Zacatecas Mexico 1557 Cuenca ( Santa Ana de los Cuatro Rios de Cuenca ) Azuay Ecuador 1557 Santo Amaro Bahia Brazil 1558 Mérida Mérida Venezuela 1558 Londres Catamarca Argentina Second oldest continuously inhabited European established settlement in Argentina , though refunded and relocated several times . 1560 Mogi das Cruzes São Paulo Brazil 1560 Ipojuca Pernambuco Brazil 1561 Santa Cruz de la Sierra Santa Cruz Bolivia 1561 Mendoza Mendoza Argentina 1561 San Cristobal Táchira Venezuela 1561 Itanhaém São Paulo Brasil 1562 San Juan San Juan Argentina 1562 Río Grande Zacatecas Mexico 1562 Susticacán Zacatecas Mexico 1563 Cartago Cartago Costa Rica Oldest continuously inhabited European established settlement in Costa Rica . 1564 São Mateus Espírito Santo Brazil 1565 Rio de Janeiro Rio de Janeiro Brazil 1565 Saint Augustine Florida United States Oldest continuously inhabited European established settlement in the continental United States . Preceded by Puerto Rican colonies , Pensacola , Florida , which was destroyed in 1559 , and Fort Caroline , destroyed in 1565 . 1565 Tucumán Tucumán Argentina 1567 Caracas Capital District Venezuela 1568 Mazapil Zacatecas Mexico 1568 Goiana Pernambuco Brazil Declared vila in 1685 ; cidade in 1840 . 1568 Valaparaíso Zacatecas Mexico 1569 Santa Ana Santa Ana El Salvador Built over Mayan city of Sihuatehuacán . 1570 Jerez Zacatecas Mexico 1570 Sombrerete Zacatecas Mexico 1571 Cochabamba Cochabamba Bolivia 1572 Huancavelica Huancavelica Peru 1573 San German Puerto Rico United States Second oldest European settlement in Puerto Rico 1573 Córdoba Córdoba Argentina 1573 Santa Fé Santa Fé Argentina 1573 Buga Valle del Cauca Colombia 1574 Huaraz Ancash Peru 1574 Tarija Tarija Department Bolivia 1575 Aguascalientes Aguascalientes Mexico 1576 León Guanajuato Mexico 1577 Juayúa Sonsonate El Salvador 1578 Fort - Liberté Nord - Est Haiti Originally called Bayaha by the Spanish . Oldest continuously inhabited European established settlement in Haiti . 1578 Tegucigalpa Francisco Morazán Honduras 1579 Coamo Puerto Rico United States Third oldest European settlement in Puerto Rico . In the same area where the Taínos had had their village of Guayama . Coamo became officially a town in 1616 , and given the title of villa by Spanish Royal Decree in 1778 . 1580 Buenos Aires Argentina 1580 Santana de Parnaíba São Paulo Brasil 1580 Santo Domingo Xenacoj Sacatepéquez Guatemala 1582 Salta Salta Argentina 1583 St John 's Newfoundland and Labrador Canada Oldest continuously inhabited British settlement in the Americas . 1584 Roanoke North Carolina United States 1585 João Pessoa Paraíba Brazil 1587 Concepción del Oro Zacatecas Mexico 1588 Corrientes Corrientes Argentina 1590 São Cristóvão Sergipe Brazil First capital of Sergipe . Formerly known as Sergipe d'El Rei 1591 Jiménez del Teul Zacatecas Mexico 1591 Juan Aldama Zacatecas Mexico 1591 Paudalho Pernambuco Brazil 1591 La Rioja La Rioja Argentina 1593 San Salvador de Jujuy Jujuy Argentina 1594 Pinos Zacatecas Mexico 1594 San Luis San Luis Argentina 1596 Monterrey Nuevo León Mexico 1596 Villahermosa Tabasco Mexico Originally called Villa Felipe II . 1597 Portobelo Colón Panama 1598 Parras Coahuila Mexico 1599 Natal Rio Grande do Norte Brazil 1599 Tadoussac Quebec Canada Oldest continuously inhabited French established settlement in the Americas , oldest European established settlement in Quebec . 1600 Facatativá Cundinamarca Colombia 1600 Zipaquirá Cundinamarca Colombia 1602 David Chiriquí Province Panama 1603 Salamanca Guanajuato Mexico 1604 Port Royal Nova Scotia Canada Oldest continuously inhabited European established settlement in Nova Scotia . 1606 Bayaguana Monte Plata Dominican Republic 1606 Ibarra Imbabura Ecuador 1606 Oruro Oruro Bolivia 1607 Jamestown Virginia United States First permanent English established settlement in the Americas . 1608 Quebec City Quebec Canada 1609 Fortaleza Ceará Brazil 1610 Kecoughtan Virginia United States 1524 León León Nicaragua Original location ( now known as the ruins of Leon Viejo ) on the south - western shore of Lago Managua founded in 1524 , but destroyed by earthquakes , and resettled 20 miles west , in 1610 , to its present - day location at Leon . 1610 Santa Fe New Mexico United States Oldest continuously inhabited state or territorial capital in the continental United States . ( 8 ) 1610 Harbour Grace Newfoundland and Labrador Canada 1611 Marechal Deodoro , Alagoas Alagoas Brazil 1612 São Luís Maranhão Brazil Capital of Maranhão . 1612 St. George 's Bermuda Oldest continuously inhabited English established settlement in the Americas . 1612 Neiva Huila Colombia 1613 Penedo Alagoas Brazil 1613 Hopewell Virginia United States Oldest continuously inhabited English settlement in North America . Founded in 1613 as Bermuda City by Thomas Dale . 1614 Albany , New York New York United States Oldest US settlement north of Virginia and second oldest state or territorial capital in the continental United States , incorporated 1686 1614 Sirinhaém Pernambuco Brazil Declared vila in 1627 ; cidade in 1892 . 1615 Cabo Frio Rio de Janeiro Brazil 1615 Taos New Mexico United States 1615 Ipiales Nariño Colombia 1616 Belém Pará Brazil Capital of Pará . 1616 Medellín Antioquia Colombia Founded by Francisco de Herrera y Campuzano 1616 Arecibo Puerto Rico United States 1617 Jersey City , New Jersey New Jersey United States Pavonia , New Netherland 1618 Cabo de Santo Agostinho Pernambuco Brazil Declared vila in 1811 ; cidade in 1877 . 1620 Ojocaliente Zacatecas Mexico 1620 Plymouth Massachusetts United States 1622 Bucaramanga Santander Colombia 1623 Dover New Hampshire United States 1623 Gloucester Massachusetts United States 1624 Old Road Saint Christopher Saint Kitts and Nevis Oldest continuously inhabited British established settlement in the Caribbean , First successful ' non-Spanish ' established settlement in the Caribbean . 1624 Villa Soriano Soriano Uruguay Oldest continuously inhabited European established settlement in Uruguay . 1625 New Amsterdam New York United States Now New York City . 1625 Quincy Massachusetts United States 1626 Salem Massachusetts United States 1626 Vitória de Santo Antão Pernambuco Brazil Declared vila in 1811 ; cidade in 1843 . 1627 Basseterre Saint Christopher Saint Kitts and Nevis Oldest continuously inhabited French established settlement in the Caribbean . 1628 Bridgetown Saint Michael Barbados Second oldest continuously inhabited English established settlement in the Caribbean . 1630 Boston Massachusetts United States 1630 Paramaribo Paramaribo Suriname Oldest continuously inhabited European established settlement in Suriname . 1630 Speightstown Saint Peter Barbados 1631 Biddeford Maine United States 1631 Hidalgo del Parral Chihuahua Mexico 1631 Saint John New Brunswick Canada 1632 Batopilas Chihuahua Mexico 1632 Williamsburg Virginia United States 1632 St. John 's Antigua Antigua and Barbuda 1634 Green Bay Wisconsin United States Oldest continuously inhabited French established settlement in the United States . 1634 St. Mary 's City Maryland United States 1634 Trois - Rivières Quebec Canada 1634 Willemstad Curaçao Netherlands Oldest continuously inhabited Dutch established settlement in the Caribbean . 1635 Concord Massachusetts United States 1635 Old Saybrook Connecticut United States Original Dutch settlement called Kievits Hoek . 1636 Springfield Massachusetts United States 1636 Providence Rhode Island United States 1637 Hartford Connecticut United States ( 9 ) 1638 Belize City Belize Belize Oldest continuously inhabited European established settlement in Belize . Original Mayan city called Holzuz . 1638 Fort - de-France Martinique Originally called Fort Saint Louis . 1638 New Haven Connecticut United States ( 10 ) 1638 Wilmington Delaware United States Grew from Fort Christina , part of the New Sweden colony . Oldest continuously inhabited Swedish established settlement in the Americas . 1638 Cambridge Massachusetts United States 1639 Rincón de Romos Aguascalientes Mexico 1639 ( prior to ) St. Marks Florida United States ( 11 ) 1639 Barnstable Massachusetts United States Second - oldest municipality and only city on Cape Cod 1639 Bridgeport Connecticut United States 1639 Guilford Connecticut United States Location of the Henry Whitfield House , the oldest house in Connecticut and the oldest stone house in New England 1639 Milford Connecticut United States 1639 Newport Rhode Island United States 1640 Tumaco Nariño Colombia 1642 Montreal Quebec Canada 1642 Lexington Massachusetts United States ( 12 ) 1642 Warwick Rhode Island United States 1643 Basse - Terre Guadeloupe France Territorial capital . 1643 Dolores Hidalgo Guanajuato Mexico 1644 Salvatierra Guanajuato Mexico 1646 New London Connecticut United States 1648 Alcântara Maranhão Brazil 1648 Paranaguá Paraná Brazil 1649 Annapolis Maryland United States 1650 Castries Castries Saint Lucia 1650 Saint George 's Saint George Grenada 1651 Norwalk Connecticut United States 1654 Vieux Fort Vieux Fort Saint Lucia 1659 Ciudad Juárez Chihuahua Mexico 1659 El Paso Texas United States 1660 Charlestown Nevis St Kitts & Nevis 1660 Placentia Newfoundland and Labrador Canada French Capital until 1713 , originally called Plaisance 1660 Rye New York United States 1664 Cayenne French Guiana France Oldest continuously inhabited European established settlement in French Guiana . 1664 Quibdó Chocó Colombia 1665 Port - de-Paix Nord - Ouest Haiti 1666 Codrington Barbuda Antigua and Barbuda 1666 Newark New Jersey United States 1667 Paraty Rio de Janeiro Brazil 1668 Sault Ste . Marie Michigan United States ( 13 ) , oldest city in Michigan . 1670 Boqueirão Paraíba Brazil 1670 Cap - Haïtien Nord Haiti Originally called `` Cap - Français '' 1670 Charleston South Carolina United States 1672 Charlotte Amalie , St. Thomas U.S. Virgin Islands United States Oldest permanent European settlement in Saint Thomas island . Oldest continuously inhabited Danish established settlement in the Americas . 1673 Kingston Ontario Canada Grew from Fort Frontenac . Continuously inhabited since 1784 . 1673 San José de Gracia Aguascalientes Mexico 1673 Worcester Massachusetts United States 1674 Cachoeira Bahia Brazil 1674 Waterbury Connecticut United States 1677 Crosswicks New Jersey United States Quaker settlement . Continuously inhabited since founding . 1680 South Orange New Jersey United States Grew from Newark ( later Orange ) . 1680 Colonia del Sacramento Colonia Uruguay 1681 Cockburn Town Turks and Caicos Islands 1681 Philadelphia Pennsylvania United States 1682 Norfolk Virginia United States 1682 São Borja Rio Grande do Sul Brazil 1683 Dover Delaware United States 1685 Escada Pernambuco Brazil 1686 Trinidad Beni Department Bolivia 1687 New Britain Connecticut United States 1687 São Luiz Gonzaga Rio Grande do Sul Brazil 1687 São Miguel das Missões Rio Grande do Sul Brazil 1687 São Nicolau Rio Grande do Sul Brazil 1689 Marigot Saint Martin 1689 San Gil Santander Colombia 1690 Port of Spain Trinidad Trinidad and Tobago 1690 São Lourenço das Missões Rio Grande do Sul Brazil 1690 Santa Rosa de Viterbo Boyacá Colombia 1692 Villanueva Zacatecas Mexico 1693 Bom Jesus da Lapa Bahia Brazil 1693 Burlington New Jersey United States 1693 Curitiba Paraná Brazil 1693 Kingston St Andrew Jamaica 1695 Nassau New Providence Bahamas 1695 Saint - Marc Artibonite Haiti 1696 Assú Rio Grande do Norte Brazil Declared vila in 1766 ; cidade in 1845 . 1696 Rimouski Quebec Canada 1698 Ambato Tungurahua Ecuador 1698 Pensacola Florida United States Known for being the first European settlement in North America . 1699 Biloxi Mississippi United States 1699 Baton Rouge Louisiana United States 1700 Barreiras Bahia Brazil 1701 Detroit Michigan United States 1701 São João del Rei Minas Gerais Brazil 1702 Jesús María Aguascalientes Mexico 1702 Mobile Alabama United States 1703 Kaskaskia Illinois United States 1705 Bath North Carolina United States 1706 Albuquerque New Mexico United States 1706 Santo Ângelo Rio Grande do Sul Brazil 1709 Chihuahua Chihuahua Mexico 1710 Chatham New Jersey United States On land purchased in 1680 . 1710 New Bern North Carolina United States Birthplace of Pepsi 1711 Beaufort South Carolina United States 1711 Ouro Preto Minas Gerais Brazil 1714 Natchitoches Louisiana United States Oldest settlement in the Louisiana Purchase 1714 Antonina Paraná Brazil 1714 Laguna Santa Catarina Brazil 1714 Serro Minas Gerais Brazil 1715 ( prior to ) Kekionga Indiana United States Capital of the Miami tribe . 1716 Natchez Mississippi United States Dates to the founding of Fort Rosalie by the French. ( 14 ) 1717 Brejo do Cruz Paraíba Brazil 1717 Zacatecas Zacatecas Mexico 1717 Westborough Massachusetts United States One Hundredth Town in Massachusetts 1718 New Orleans Louisiana United States 1718 Rio de Contas Bahia Brazil 1718 San Antonio Texas United States 1718 Tiradentes Minas Gerais Brazil 1719 Trenton New Jersey United States 1721 Cortazar Guanajuato Mexico 1722 Edenton North Carolina United States 1723 Guaynabo Puerto Rico United States Declared pueblo in 1723 ; municipio in 1768 . 1724 Brattleboro Vermont United States Grew out of Fort Dummer . 1725 Concord New Hampshire United States 1726 Florianópolis Santa Catarina Brazil 1726 Montevideo Montevideo Uruguay 1727 Cuiabá Mato Grosso Brazil 1727 Goiás Goiás Brazil 1727 Pirenópolis Goiás Brazil 1728 Fredericksburg Virginia United States 1728 Inhambupe Bahia Brazil 1728 Nuuk Greenland Norse colony was originally called Godthab . Oldest continuously inhabited European established settlement in Greenland . 1729 Baltimore Maryland United States 1730 Roseau Saint George Dominica 1732 Fredericton New Brunswick Canada Originally called Ste . Anne 's Point . 1732 Vincennes Indiana United States 1733 Morretes Paraná Brazil 1733 Pau dos Ferros Rio Grande do Norte Brazil 1733 Richmond Virginia United States ( 15 ) 1733 Savannah Georgia United States 1733 Cúcuta Norte de Santander Colombia 1736 Augusta Georgia United States 1737 Rio Grande Rio Grande do Sul Brazil 1738 Fort Rouge Manitoba Canada Now Winnipeg , Manitoba . 1738 San José San José Costa Rica 1740 Santa Rosalía de Camargo Chihuahua Mexico 1740 Portalegre Rio Grande do Norte Brazil 1740 San Felipe de Puerto Plata Puerto Plata Dominican Republic 1741 Viamão Rio Grande do Sul Brazil 1743 Canguaretama Rio Grande do Norte Brazil Declared vila in 1858 ; cidade in 1885 . 1748 Caicó Rio Grande do Norte Brazil Declared vila in 1788 ; cidade in 1868 . 1749 Alexandria Virginia United States 1749 San Ignacio de Velasco Santa Cruz Department Bolivia 1749 Port - au - Prince Ouest Haiti 1749 Halifax Nova Scotia Canada 1749 Windsor Ontario Canada Oldest continually - inhabited settlement in Canada west of Montreal 1751 Manchester New Hampshire United States Former American textile capital - Named after Manchester , England 1752 Buíque Pernambuco Brazil 1752 Patos Paraíba Brazil Declared vila in 1832 ; cidade in 1903 . 1752 Tubac Arizona United States 1754 Augusta Maine United States 1755 Charlotte North Carolina United States 1756 Salto Salto Uruguay 1756 Santa Bárbara de Samaná Samaná Dominican Republic 1756 Yauco Puerto Rico United States 1757 Estância Sergipe Brazil 1758 Pittsburgh Pennsylvania United States 1758 Macapá Amapá Brazil 1760 Arês Rio Grande do Norte Brazil 1760 Sabana de la Mar Hato Mayor Dominican Republic 1761 Sacaba Cochabamba Department Bolivia 1762 Kingstown St Vincent Saint Vincent and the Grenadines 1762 Parnaíba Piauí Brazil 1762 Shepherdstown West Virginia United States Originally known as Mecklenburg . 1762 Allentown Pennsylvania United States Incorporated as Northamptontown . 1762 Baía da Traição Paraíba Brazil 1763 Burlington Vermont United States 1763 Pánfilo Natera Zacatecas Mexico 1763 Ellsworth Maine United States 1763 Philipsburg Sint Maarten Netherlands 1763 St. Louis Missouri United States 1764 Charlottetown Prince Edward Island Canada 1764 Pointe - à - Pitre Grand Terre Guadeloupe 1765 Port Elizabeth Grenadines Saint Vincent and the Grenadines 1765 Portsmouth Saint John Dominica 1767 Ceará - Mirim Rio Grande do Norte Brazil 1767 Araioses Maranhão Brazil 1769 Elizabethton Tennessee United States 1769 Falmouth Trelawny Jamaica 1769 San Diego California United States Grew from Presidio of San Diego 1769 Lapa Paraná Brazil 1770 Monterey California United States Grew from Presidio of Monterey . Original capital of California ( 16 ) 1770 Araxá Minas Gerais Brazil 1771 Calvillo Aguascalientes Mexico 1772 Porto Alegre Rio Grande do Sul Brazil 1775 Tucson Arizona United States 1774 Castro Paraná Brazil 1775 Lexington Kentucky United States 1775 Boonesborough Kentucky United States Grew from Fort Boonesborough , built by pioneer Daniel Boone . - 1775 São Lourenço da Mata Pernambuco Brazil District status in 1775 ; village status in 1884 ; municipality status in 1890 1776 San Francisco California United States 1776 Sincelejo Sucre Colombia 1777 San Jose California United States Originally known as El Pueblo de San José de Guadalupe , the first town in the Spanish colony of Nueva California , which later became Alta California . 1777 Montería Cordoba Colombia 1778 Corumbá Mato Grosso do Sul Brazil 1778 Louisville Kentucky United States Grew from Fort Nelson , established by explorer George Rogers Clark . 1778 San Francisco de Macorís Duarte Dominican Republic 1779 Jonesborough Tennessee United States Later organized as the lost State of Franklin with Jonesborough as capital 1784 . 1779 Nashville Tennessee United States Grew from Fort Nashborough 1780 Codó Maranhão Brazil Declared vila in 1833 ; cidade in 1896 . 1780 Las Matas de Farfán San Juan Dominican Republic 1780 Arauca Arauca Colombia 1781 Montpelier Vermont United States 1781 Los Angeles California United States 1782 Catu Bahia Brazil 1782 Georgetown Demerara - Mahaica Guyana Originally called La Nouvelle Ville . 1783 Vitória da Conquista Bahia Brazil 1784 San Fernando Trinidad Trinidad and Tobago 1785 Harrisburg Pennsylvania United States 1785 North Battleford Saskatchewan Canada 1785 fur trading post . 1877 Battleford is capital of the North West Territories . North Battleford incorporated as a city ( population 5000 + ) 1913 . 1785 Asheville North Carolina United States 1785 Gustavia Saint Barthélemy 1785 Huntington West Virginia United States 1785 Sydney Nova Scotia Canada 1786 Columbia South Carolina United States 1786 Escada Pernambuco Brazil Declared vila in 1854 ; cidade in 1873 . 1786 Florissant Missouri United States Originally known as St. Ferdinand . 1786 Frankfort Kentucky United States 1786 Portland Maine United States 1787 Boa Ventura Paraíba Brazil 1788 Marietta Ohio United States 1788 Cincinnati Ohio United States 1788 Charleston West Virginia United States Grew from Fort Lee . ( 17 ) 1788 Mercedes Soriano Uruguay 1789 Buffalo New York United States 1789 Itambé Pernambuco Brazil Declared vila in 1867 ; cidade in 1879 . 1790 Hamilton Bermuda 1790 Washington District of Columbia United States 1791 Jacksonville Florida United States Known as `` Cowford '' . 1791 Knoxville Tennessee United States 1791 Bangor Maine United States 1792 Raleigh North Carolina United States 1793 Sherbrooke Quebec Canada 1793 Toronto Ontario Canada 1794 Fort Wayne Indiana United States 1795 Edmonton Alberta Canada Grew from Fort Edmonton . 1795 Erie Pennsylvania United States Grew from the French Fort Presque Isle . 1795 Maryville Tennessee United States Grew from the American Fort Craig . 1796 Cleveland Ohio United States 1796 Oranjestad Aruba Netherlands 1797 Chaguanas Trinidad Trinidad and Tobago 1797 Franklinton Ohio United States Eventually absorbed by Columbus , Ohio . 1799 Araruama Rio de Janeiro Brazil 1800 Hull Quebec Canada Formerly known as Wrightville . 1800 Sonsón Antioquia Colombia 1802 Chapadinha Maranhão Brazil 1802 Santa Cruz de Barahona Barahona Dominican Republic 1803 Chicago Illinois United States Grew from Fort Dearborn . 1805 Huntsville Alabama United States 1807 Hot Springs Arkansas United States 1809 Água Preta Pernambuco Brazil Declared vila in 1846 ; cidade in 1895 . 1810 Guarapuava Paraná Brazil 1810 Manchester New Hampshire United States 1810 Pilão Arcado Bahia Brazil 1810 San Bernardino California United States 1811 Garanhuns Pernambuco Brazil 1811 Astoria Oregon United States 1812 Columbus Ohio United States 1812 Pelotas Rio Grande do Sul Brazil Declared vila in 1832 ; cidade in 1835 . 1813 Capela Sergipe Brazil 1813 Chisec Alta Verapaz Guatemala 1813 Barranquilla Atlántico Colombia 1815 Areia Paraíba Brazil 1815 Hamilton Ontario Canada 1816 Alagoinhas Bahia Brazil 1816 Chattanooga Tennessee United States Originally named Ross 's Landing . 1816 Saginaw Michigan United States 1816 Savaneta Aruba Netherlands 1817 Fort Smith Arkansas United States 1818 Itaguaí Rio de Janeiro Brazil 1819 Montgomery Alabama United States Near the site of the earlier French Fort Toulouse . 1819 Springfield Illinois United States 1819 Managua Managua Nicaragua 1819 Memphis Tennessee United States Near the site of the earlier French Fort Prudhomme . 1819 Palmeira Paraná Brazil 1819 Tuscaloosa Alabama United States Former state capital , grew from former Native American settlements beginning in 1580 . ( 18 ) 1820 Cotegipe Bahia Brazil 1821 Little Rock Arkansas United States 1821 Indianapolis Indiana United States 1821 Jefferson City Missouri United States 1821 Decatur Alabama United States 1822 Ponta Grossa Paraná Brazil 1822 Jackson Mississippi United States 1823 Tampa Florida United States Grew from earlier military post Fort Brooke . 1824 Vancouver Washington United States 1824 Tallahassee Florida United States 1825 Vicksburg Mississippi United States 1825 Grand Rapids Michigan United States 1825 Syracuse New York United States incorporated as village ; received city charter in 1847 1826 Ipiranga Paraná Brazil 1826 London Ontario Canada 1826 Bytown Ontario Canada Now Ottawa , Ontario . 1827 Fort Leavenworth Kansas United States 1827 St. Andrews Florida United States Now part of Panama City 1831 Greenville South Carolina United States 1831 Piracanjuba Goiás Brazil 1832 Goianinha Rio Grande do Norte Brazil 1828 Key West Florida United States 1832 Touros Rio Grande do Norte Brazil 1833 Bananeiras Paraíba Brazil 1833 Juazeiro Bahia Brazil 1833 Kitchener Ontario Canada Originally called Berlin . 1833 Milwaukee Wisconsin United States 1833 Nazaré da Mata Pernambuco Brazil Declared vila in 1833 ; cidade in 1850 . 1833 Rio Formoso Pernambuco Brazil Declared vila in 1833 ; cidade in 1850 . 1833 Vassouras Rio de Janeiro Brazil 1835 Austin Texas United States 1835 Barra do Corda Maranhão Brazil 1835 Curupuru Maranhão Brazil Declared vila in 1841 ; cidade in 1920 . 1836 Davenport Iowa United States 1836 Oshawa Ontario Canada 1836 Shreveport Louisiana United States 1836 Madison Wisconsin United States 1836 Tulsa Oklahoma United States The Lochapoka named their new settlement `` Tulasi , '' meaning `` old town '' 1837 Glória do Goitá Pernambuco Brazil Declared vila in 1877 ; cidade in 1884 . 1837 Lansing Michigan United States 1837 Houston Texas United States 1838 Divina Pastora Sergipe Brazil 1838 Kansas City Missouri United States 1838 St Paul Minnesota United States 1839 Santa Rita Paraíba Brazil Declared vila in 1890 ; cidade in 1924 . 1840 Summerside Prince Edward Island Canada 1840 Villavicencio Meta Colombia 1841 Cedar Rapids Iowa United States 1841 Dallas Texas United States 1842 Piatã Bahia Brazil 1842 Salem Oregon United States 1843 Atlanta Georgia United States Originally known as Terminus , later renamed Marthasville . Acquired the name Atlanta in 1847 . 1843 Des Moines Iowa United States 1843 Stanley East Falkland Falkland Islands 1843 Uruguaiana Rio Grande do Sul Brazil Declared vila in 1846 ; cidade in 1874 . 1843 Victoria British Columbia Canada 1844 Lençóis Bahia Brazil 1845 Araruna Paraíba Brazil 1845 Portland Oregon United States 1847 Alagoa Grande Paraíba Brazil Declared vila in 1865 ; cidade in 1908 . 1847 Salt Lake City Utah United States Originally known as Great Salt Lake City . 1848 Caruaru Pernambuco Brazil First municipality in the ( Agreste ) region of Pernambuco state , Brazil . 1848 Rio Verde Goiás Brazil 1848 Sacramento California United States 1849 Provo Utah United States 1849 Manizales Caldas Colombia 1851 San Luis Colorado United States 1851 Seattle Washington United States 1851 Jericó Antioquia Colombia 1852 Mossoró Rio Grande do Norte Brazil 1852 São José dos Pinhais Paraná Brazil 1852 Colón Colón Panama 1853 Olympia Washington United States 1854 Omaha Nebraska United States 1854 Anajatuba Maranhão Brazil 1854 Topeka Kansas United States 1855 Aracaju Sergipe Brazil 1856 Lincoln Nebraska United States Originally called Lancaster . 1856 O'Fallon Missouri United States 1856 Sioux Falls South Dakota United States 1857 Boquim Sergipe Brazil 1858 Carson City Nevada United States 1858 Denver Colorado United States 1859 Yankton South Dakota United States 1860 Chico California United States 1863 Boise Idaho United States 1863 Wichita Kansas United States 1864 Casper Wyoming United States Originally called Platte Bridge Station . 1864 Helena Montana United States 1864 Salinas California United States ( 19 ) 1866 Prince Albert Saskatchewan Canada Settled by Rev. James Nisbet . Incorporated as city ( population 5000 + ) October 8 , 1904 1867 Cheyenne Wyoming United States 1867 Gameleira Pernambuco Brazil 1867 Minneapolis Minnesota United States 1868 Colinas Maranhão Brazil Declared vila in 1870 ; cidade in 1891 . 1868 Phoenix Arizona United States 1870 Afogados da Ingazeira Pernambuco Brazil 1870 Campo Largo Paraná Brazil 1870 Puerto Limon Limon Costa Rica 1871 Águas Belas Pernambuco Brazil 1871 Alto Parnaíba Maranhão Brazil 1871 Birmingham Alabama United States 1871 Colorado Springs Colorado United States 1871 Fargo North Dakota United States 1871 Imbituva Paraná Brazil 1871 São João do Triunfo Paraná Brazil 1872 Anniston Alabama United States 1872 Bismarck North Dakota United States 1872 Tibagi Paraná Brazil 1873 Jaboatão dos Guararapes Pernambuco Brazil First settlement in 1593 ; village status in 1873 ; city status in 1884 1873 Timbaúba Pernambuco Brazil Declared vila in 1879 ; cidade in 1884 . 1875 Orlando Florida United States 1875 Calgary Alberta Canada Grew from Fort Calgary 1875 Humboldt Saskatchewan Canada Incorporated as a city ( population 5000 + ) November 7 , 2000 1876 Boa Vista Paraíba Brazil 1876 Cajapió Maranhão Brazil 1876 Rapid City South Dakota United States 1877 Billings Montana United States 1877 Jaquimeyes Barahona Dominican Republic 1878 Coeur d'Alene Idaho United States 1878 Colombo Paraná Brazil 1879 Vicência Pernambuco Brazil Declared vila in 1891 ; cidade in 1928 . 1880 Pierre South Dakota United States 1881 Juneau Alaska United States 1881 Moose Jaw Saskatchewan Canada Incorporated as a city ( population 5000 + ) November 1903 . 1882 Brandon Manitoba Canada 1882 Mao Valverde Dominican Republic 1882 Restauración Dajabón Dominican Republic 1882 Yorkton Saskatchewan Canada The York Farmers ' Colonization Company Limited settlement May 12 , 1882 . City ( population 5000 + ) incorporated February 1 , 1928 . 1882 Swift Current Saskatchewan Canada Canadian Pacific Railway arrival December 10 , 1882 . City ( population 5000 + ) incorporated January 15 , 1914 . 1882 Regina Saskatchewan Canada Settlement first named Pile O ' Bones , and renamed Regina . Capital city of SK . Incorporated as a city ( population 5000 + ) June 19 , 1903 1882 Rio Branco Acre Brazil 1883 Saskatoon Saskatchewan Canada Temperance Colony founded at Nutana . Riversdale , Nutana , and Saskatoon incorporated as a city ( population 5000 + ) in 1906 . 1884 Melfort Saskatchewan Canada Stoney Creek Settlement began in 1884 . Incorporated as a city ( population 5000 + ) on July 1 , 1907 . 1886 Vancouver British Columbia Canada 1887 Gulfport Mississippi United States ( 20 ) 1888 Virginia Beach Virginia United States Grew from Seatack community . 1889 Oklahoma City Oklahoma United States 1889 Whiting Indiana United States 1890 Ijuí Rio Grande do Sul Brazil 1892 Aliança Pernambuco Brazil Declared vila in 1909 ; cidade in 1928 . 1892 Catende Pernambuco Brazil Created in 1892 as a district of Palmares . Autonomous municipality in 1909 . 1892 Estevan Saskatchewan Canada CPR arrives 1893 . Incorporated as city ( population 5000 + ) March 1 , 1957 1892 Guayaramerín Beni Bolivia 1892 Restauración Dajabón Dominican Republic 1892 Vernon British Columbia Canada 1893 Araripina Pernambuco Brazil Created in 1893 as a district of Ouricuri . Autonomous municipality in 1928 . 1893 Catuípe Rio Grande do Sul Brazil 1894 Riberalta Beni Bolivia 1895 Traverse City Michigan United States 1896 Condado Pernambuco Brazil Declared cidade in 1958 . 1896 Inhumas Goiás Brazil 1896 Miami Florida United States 1896 Santa Catalina la Tinta Alta Verapaz Guatemala 1897 Jequié Bahia Brazil 1897 Nelson British Columbia Canada 1898 Weyburn Saskatchewan Canada Signal Hill settlement . Weyburn incorporated as a city ( population 5000 + ) 1913 . 1898 Chetumal Quintana Roo Mexico Originally called Payo Obispo . 1899 Porto Acre Acre Brazil 1900 Afrânio Pernambuco Brazil Created in 1900 as a district of Petrolina . Autonomous municipality in 1963 . 1901 Fairbanks Alaska United States 1902 Florencia Caquetá Colombia 1903 Lloydminster Saskatchewan Canada Brittania colony settled by Rev. Exton Lloyd and Rev. Isaac Barr . Incorporated as a city ( population 5000 + ) in 1958 . 1904 Cruzeiro do Sul Acre Brazil 1904 Grytviken South Georgia 1904 Sena Madureira Acre Brazil 1904 Xapuri Acre Brazil 1906 Cobija Pando Bolivia 1906 Feijó Acre Brazil 1907 Melville Saskatchewan Canada Incorporated as a city ( population 5000 + ) August 1 , 1960 1907 Tarauacá Acre Brazil 1908 São Mateus do Sul Paraná Brazil 1909 Estreito Maranhão Brazil 1910 Brasiléia Acre Brazil 1910 Kindersley Saskatchewan Canada 1910 Porto Walter Acre Brazil 1910 Villazón Potosí Bolivia 1911 Las Vegas Nevada United States 1914 Anchorage Alaska United States 1915 Yopal Casanare Colombia 1928 Leticia Amazonas Colombia 1929 Pabellón de Arteaga Aguascalientes Mexico 1930 Puerto Carreño Vichada Colombia 1931 Loreto Zacatecas Mexico 1935 Yellowknife Northwest Territories Canada 1935 Mitú Vaupés Colombia 1938 San José del Guaviare Guaviare Colombia 1942 Iqaluit Nunavut Canada Originally called Frobisher Bay . 1955 Freeport Grand Bahama Bahamas 1957 Ciudad del Este Alto Paraná Paraguay Originally called Puerto Flor de Lis . 1957 Rankin Inlet Nunavut Canada 1960 Brasília Distrito Federal Brazil Created in 1960 as the national capital . 1970 Belmopan Cayo Belize 1970 Linden Upper Demerara - Berbice Guyana City formed by combining the towns of Christianburg , MacKenzie , and Wismar . 1970 Cancún Quintana Roo Mexico Inírida Guainía Colombia Palmas Tocantins Brazil Was founded 1 year after the creation of the State of Tocantins . It 's the newest and smallest Brazilian capital city .
  Fact 0.0: Tlapacoya (Ixtapaluca, Mexico State, Mexico) was founded in 7500 BC.
  Fact 0.1: Puerto Hormiga Culture (Cartagena, Colombia) was founded in 4000 BC.
  Fact 0.2: Aspero (Norte Chico, Peru) was founded in 3710 BC.
  Fact 0.3: Caral (Norte Chico, Peru) was founded in 2627 BC.
  Fact 0.4: Kaminaljuyu (Guatemala city, Guatemala) was founded in 1500 BC.
  Fact 0.5: Cuicuilco (Mexico City, Mexico) was founded in 800 BC.
  Fact 0.6: Ticul (Yucatan, Mexico) was founded in 700 BC.
  Fact 0.7: Cholula (Puebla, Mexico) was founded in 500 BC.
  Fact 0.8: Teotihuacan (Mexico State, Mexico) was founded in 300 BC.
  Fact 0.9: Teotihuacan is in the Valley of Mexico.
  Fact 0.10: Mitla (Oaxaca, Mexico) was founded in 200.
  Fact 0.11: Acanceh (Yucatan, Mexico) was founded in 200.
  Fact 0.12: Cuenca (Azuay, Ecuador) was founded in 500.
  Fact 0.13: Ejutla (Oaxaca, Mexico) was founded in 524.
  Fact 0.14: Izamal (Yucatan, Mexico) was founded in 550.
  Fact 0.15: Cahokia (Illinois, United States) was founded in 600.
  Fact 0.16: Tula (Hidalgo, Mexico) was founded in 713.
  Fact 0.17: L'Anse aux Meadows (Newfoundland and Labrador, Canada) was founded in 1001.
  Fact 0.18: L'Anse aux Meadows is the oldest known European (Norse) settlement in the Americas.
  Fact 0.19: L'Anse aux Meadows was possibly founded by Leif Erikson as part of Vinland in 1001 AD.
  Fact 0.20: Acoma Pueblo and Taos Pueblo (New Mexico, United States) were founded in 1000.
  Fact 0.21: Motul (Yucatan, Mexico) was founded in 1050.
  Fact 0.22: Pachuca (Hidalgo, Mexico) was founded in 1050.
  Fact 0.23: Cusco (Cusco Province, Peru) was founded in 1100.
  Fact 0.24: Oraibi (Arizona, United States, Hopi Reservation) was founded in 1100.
  Fact 0.25: Ecatepec (Estado de Mexico, Mexico) was founded in 1168.
  Fact 0.26: Chimaulhuacan (Mexico State, Mexico) was founded in 1259.
  Fact 0.27: Xalapa (Veracruz, Mexico) was founded in 1313.
  Fact 0.28: Tenochtitlan (Mexico City, Mexico) was founded in 1325.
  Fact 0.29: Tenochtitlan was the largest pre-Columbian city in the Americas.
  Fact 0.30: Tenochtitlan was later called Mexico City.
  Fact 0.31: Zuni Pueblo (New Mexico, United States) was founded in 1450.
  Fact 0.32: Iximche (Chimaltenango, Guatemala) was founded in 1470.
  Fact 0.33: La Navidad (Nord-Est, Haiti) was founded in 1492.
  Fact 0.34: La Navidad was the first European settlement in the New World during the Age of Discovery.
  Fact 0.35: La Isabela (Puerto Plata, Dominican Republic) was founded in 1493.
  Fact 0.36: Santo Domingo (Distrito Nacional, Dominican Republic) was founded in 1498.
  Fact 0.37: Santo Domingo is the capital of the Dominican Republic.
  Fact 0.38: Santo Domingo is the oldest continuously inhabited European established settlement in the Americas.
  Fact 0.39: Santo Domingo was founded in 1498 by Bartholomew Columbus.
  Fact 0.40: Santa Cruz del Seibo (El Seibo, Dominican Republic) was founded in 1502.
  Fact 0.41: Azua de Compostela (Azua, Dominican Republic) was founded in 1504.
  Fact 0.42: Santiago de los Caballeros (Santiago, Dominican Republic) was founded in 1504.
  Fact 0.43: Cotui (Sanchez Ramirez, Dominican Republic) was founded in 1505.
  Fact 0.44: Salvaleon de Higuey (La Altagracia, Dominican Republic) was founded in 1506.
  Fact 0.45: Bonao (Monseñor Nouel, Dominican Republic) was founded in 1508.
  Fact 0.46: Caparra (Puerto Rico, United States) was founded in 1508.
  Fact 0.47: Caparra was the first city founded by Europeans in the United States, although not continuously inhabited.
  Fact 0.48: Caparra was abandoned in 1521 with the removal of the capital to San Juan.
  Fact 0.49: Santa Maria la Antigua del Darien (Uraba, Colombia) was founded in 1510.
  Fact 0.50: Santa Maria la Antigua del Darien was the first city founded by Europeans on the South American continent.
  Fact 0.51: Nombre de Dios (Colon, Panama) was founded in 1510.
  Fact 0.52: Nombre de Dios is the oldest continuously inhabited European established settlement in Panama.
  Fact 0.53: Baracoa (Guantanamo, Cuba) was founded in 1511.
  Fact 0.54: Baracoa is the oldest continuously inhabited European established settlement in Cuba.
  Fact 0.55: Baracoa is a former capital of Cuba.
  Fact 0.56: Bayamo (Granma, Cuba) was founded in 1513.
  Fact 0.57: Sancti Spiritus (Santi Spiritus, Cuba) was founded in 1514.
  Fact 0.58: Trinidad (Santi Spiritus, Cuba) was founded in 1514.
  Fact 0.59: Santiago (Santiago, Cuba) was founded in 1515.
  Fact 0.60: Santiago was the second capital of Cuba, after Baracoa and before Havana.
  Fact 0.61: Havana (Havana, Cuba) was founded in 1519.
  Fact 0.62: La Villa Rica de la Vera Cruz (Veracruz, Mexico) was founded in 1519.
  Fact 0.63: La Villa Rica de la Vera Cruz is the oldest continuously inhabited European established settlement in Mexico.
  Fact 0.64: Panama City (Panama, Panama) was founded in 1519.
  Fact 0.65: Panama City was the first European established city on the Pacific Coast of the Americas.
  Fact 0.66: Panama City was founded at the present day ruins of Panama Viejo.
  Fact 0.67: Panama City was sacked by the Welsh pirate Henry Morgan in 1671.
  Fact 0.68: Panama City was resettled to nearby Casco Viejo in 1673.
  Fact 0.69: Hato Mayor del Rey (Hato Mayor, Dominican Republic) was founded in 1520.
  Fact 0.70: Cumana (Sucre, Venezuela) was founded in 1515.
  Fact 0.71: Cumana is the oldest continuously inhabited European established settlement in Venezuela.
  Fact 0.72: San Juan (Puerto Rico, United States) was founded in 1521.
  Fact 0.73: San Juan is the oldest continuously inhabited European established settlement in Puerto Rico.
  Fact 0.74: San Juan is the oldest continuously inhabited European established settlement in the United States.
  Fact 0.75: San Juan is the oldest continuously inhabited state or territorial capital in the United States.
  Fact 0.76: Granada (Granada, Nicaragua) was founded in 1524.
  Fact 0.77: Granada is the oldest continuously inhabited European established settlement in Nicaragua.
  Fact 0.78: Tecpan Guatemala (Chimaltenango, Guatemala) was founded in 1524.
  Fact 0.79: Tecpan Guatemala was the first capital of Guatemala.
  Fact 0.80: Tecpan Guatemala is the oldest continuously inhabited European established settlement in Guatemala.
  Fact 0.81: Trujillo (Colon, Honduras) was founded in 1524.
  Fact 0.82: Trujillo is the oldest continuously inhabited European established settlement in Honduras.
  Fact 0.83: Santa Marta (Magdalena, Colombia) was founded in 1525.
  Fact 0.84: Santa Marta is the oldest continuously inhabited European established settlement in Colombia.
  Fact 0.85: Acapulco (Guerrero, Mexico) was founded in 1525.
  Fact 0.86: San Salvador (San Salvador, El Salvador) was founded in 1525.
  Fact 0.87: San Salvador was rebuilt and changed locations twice afterwards, in 1528 and 1545.
  Fact 0.88: San Salvador is the oldest continuously inhabited European established settlement in El Salvador.
  Fact 0.89: Acambaro (Guanajuato, Mexico) was founded in 1526.
  Fact 0.90: Chimaltenango (Chimaltenango, Guatemala) was founded in 1526.
  Fact 0.91: San Cristobal de las Casas (Chiapas, Mexico) was founded in 1528.
  Fact 0.92: San Cristobal de las Casas was originally called Villa Real de Chiapa.
  Fact 0.93: Camaguey (Camaguey Province, Cuba) was founded in 1528.
  Fact 0.94: Camaguey was founded in 1515 on the northern coast as Puerto del Principe.
  Fact 0.95: Camaguey was moved in 1528 to its present location.
  Fact 0.96: Maracaibo (Zulia, Venezuela) was founded in 1529.
  Fact 0.97: San Blas (Nayarit, Mexico) was founded in 1530 by Nuño Beltrán.
  Fact 0.98: Puebla (Puebla, Mexico) was founded in 1531 by Fr. Toribio de Benavente (Motolinia) on 16 April.
  Fact 0.99: Oaxaca (Oaxaca, Mexico) was founded in 1532.
  Fact 0.100: Piura (Piura, Peru) was founded in 1532.
  Fact 0.101: Sao Vicente (Sao Paulo, Brazil) was founded in 1532.
  Fact 0.102: Sao Vicente is the oldest continuously inhabited European established settlement in Brazil.
  Fact 0.103: Sao Vicente is the oldest Portuguese established settlement in the New World.
  Fact 0.104: Tepic (Nayarit, Mexico) was founded in 1532 as capital of Nueva Galicia.
  Fact 0.105: Cananeia (Sao Paulo, Brazil) was founded in 1533.
  Fact 0.106: Cananeia is the second oldest city in Brazil.
  Fact 0.107: Cartagena de Indias (Bolivar, Colombia) was founded in 1533.
  Fact 0.108: Otavalo (Imbabura, Ecuador) was founded in 1534.
  Fact 0.109: Quito (Pichincha, Ecuador) was founded in 1534.
  Fact 0.110: Quito is the oldest continuously inhabited European established settlement in Ecuador.
  Fact 0.111: Spanish Town (Saint Catherine, Jamaica) was founded in 1534 by the Spanish as Santiago de la Vega or St. Jago de la Vega.
  Fact 0.112: Spanish Town is the oldest continuously inhabited European established settlement in Jamaica.
  Fact 0.113: Trujillo (La Libertad, Peru) was founded in 1534.
  Fact 0.114: Trujillo is the oldest continuously inhabited European established settlement in Peru.
  Fact 0.115: Lima (Lima, Peru) was founded in 1535.
  Fact 0.116: Igarassu (Pernambuco, Brazil) was the site of the first European settlement in Brazil, the feitoria of Igarassu, in 1516.
  Fact 0.117: Igarassu was founded in 1535.
  Fact 0.118: Olinda (Pernambuco, Brazil) was founded in 1535.
  Fact 0.119: Olinda is one of the best-preserved colonial cities in Brazil.
  Fact 0.120: Vila Velha (Espirito Santo, Brazil) was founded in 1535.
  Fact 0.121: Paria (Oruro, Bolivia) was founded in 1535.
  Fact 0.122: Paria was the first Spanish settlement in Bolivia.
  Fact 0.123: Santiago de Cali (Valle del Cauca, Colombia) was founded in 1536.
  Fact 0.124: Popayan (Cauca, Colombia) was founded in 1536.
  Fact 0.125: San Pedro Sula (Cortes, Honduras) was founded in 1536.
  Fact 0.126: Tupiza (Potosi, Bolivia) was founded in 1536.
  Fact 0.127: Asuncion (Paraguay) was founded in 1537.
  Fact 0.128: Asuncion is the oldest continuously inhabited European established settlement in Paraguay.
  Fact 0.129: Recife (Pernambuco, Brazil) was founded in 1537.
  Fact 0.130: Recife is the capital of the state of Pernambuco.
  Fact 0.131: Recife is the oldest city among Brazil's current state capitals.
  Fact 0.132: Bogota (Cundinamarca, Colombia) was founded in 1538.
  Fact 0.133: Guayaquil (Guayas, Ecuador) was founded in 1538.
  Fact 0.134: Iguape (Sao Paulo, Brazil) was founded in 1538.
  Fact 0.135: Sucre (Chuquisaca, Bolivia) was founded in 1538.
  Fact 0.136: Sucre was originally called Ciudad de la Plata de la Nueva Toledo.
  Fact 0.137: Sucre is the oldest continuously inhabited European established settlement in Bolivia.
  Fact 0.138: Tunja (Boyaca, Colombia) was founded in 1539.
  Fact 0.139: Velez (Santander, Colombia) was founded in 1539.
  Fact 0.140: Pasto (Narino, Colombia) was founded in 1539.
  Fact 0.141: Arequipa (Arequipa, Peru) was founded in 1540.
  Fact 0.142: Ayacucho (Ayacucho, Peru) was founded in 1540.
  Fact 0.143: Campeche (Campeche, Mexico) was founded in 1540.
  Fact 0.144: Yuriria (Guanajuato, Mexico) was founded in 1540.
  Fact 0.145: Mompox (Bolivar, Colombia) was founded in 1540.
  Fact 0.146: Supia (Caldas, Colombia) was founded in 1540.
  Fact 0.147: Campeche was founded on the site of the Mayan village Kimpech.
  Fact 0.148: Huanuco (Huanuco, Peru) was founded in 1541.
  Fact 0.149: Moquegua (Moquegua, Peru) was founded in 1541.
  Fact 0.150: Santiago (Santiago, Chile) was founded in 1541.
  Fact 0.151: Santiago is the oldest continuously inhabited European established settlement in Chile.
  Fact 0.152: Santafe de Antioquia (Antioquia, Colombia) was founded in 1541.
  Fact 0.153: Guadalajara (Jalisco, Mexico) was founded in 1542.
  Fact 0.154: Genaro Codina (Zacatecas, Mexico) was founded in 1542.
  Fact 0.155: San Miguel de Allende (Guanajuato, Mexico) was founded in 1542.
  Fact 0.156: Merida (Yucatan, Mexico) was founded in 1542.
  Fact 0.157: Merida was built on the site of the Mayan city T'ho.
  Fact 0.158: San Luis de la Paz (Guanajuato, Mexico) was founded in 1542.
  Fact 0.159: Antigua Guatemala (Sacatepequez, Guatemala) was founded in 1543.
  Fact 0.160: Coban (Alta Verapaz, Guatemala) was founded in 1543.
  Fact 0.161: Santa Cruz Verapaz (Alta Verapaz, Guatemala) was founded in 1543.
  Fact 0.162: Potosi (Potosi, Bolivia) was founded in 1545.
  Fact 0.163: San Juan de los Remedios (Villa Clara Province, Cuba) was founded in 1545.
  Fact 0.164: San Juan de los Remedios has the most uncertain foundation date in Cuba.
  Fact 0.165: The date 1545 for San Juan de los Remedios was accepted because it is when the village erected its town hall.
  Fact 0.166: The origin of San Juan de los Remedios has been documented back to 1513.
  Fact 0.167: Riohacha (La Guajira, Colombia) was founded in 1545.
  Fact 0.168: Santos (Sao Paulo, Brazil) was founded in 1546.
  Fact 0.169: Neyba (Bahoruco, Dominican Republic) was founded in 1546.
  Fact 0.170: Tepezala (Aguascalientes, Mexico) was founded in 1546.
  Fact 0.171: Asientos (Aguascalientes, Mexico) was founded in 1548.
  Fact 0.172: La Paz (La Paz, Bolivia) was founded in 1548.
  Fact 0.173: Panuco (Zacatecas, Mexico) was founded in 1548.
  Fact 0.174: Zacatecas (Zacatecas, Mexico) was founded in 1548.
  Fact 0.175: Salvador (Bahia, Brazil) was founded in 1549.
  Fact 0.176: Concepcion (Concepcion, Chile) was founded in 1550.
  Fact 0.177: Valledupar (Cesar, Colombia) was founded in 1550.
  Fact 0.178: Ibague (Tolima, Colombia) was founded in 1550.
  Fact 0.179: Vitoria (Espirito Santo, Brazil) was founded in 1551.
  Fact 0.180: Mocoa (Putumayo) was founded in 1551.
  Fact 0.181: Sonsonate (Sonsonate, El Salvador) was founded in 1552.
  Fact 0.182: Santiago del Estero (Santiago del Estero, Argentina) was founded in 1553.
  Fact 0.183: Santiago del Estero is the oldest continuously inhabited European established settlement in Argentina.
  Fact 0.184: Sao Bernardo do Campo (Sao Paulo, Brazil) was founded in 1553.
  Fact 0.185: Embu (Sao Paulo, Brazil) was founded in 1554.
  Fact 0.186: Sao Paulo (Sao Paulo, Brazil) was founded in 1554.
  Fact 0.187: Sain Alto (Zacatecas, Mexico) was founded in 1555.
  Fact 0.188: Chalchihuites (Zacatecas, Mexico) was founded in 1556.
  Fact 0.189: Cuenca (Azuay, Ecuador) was founded in 1557 as Santa Ana de los Cuatro Rios de Cuenca.
  Fact 0.190: Santo Amaro (Bahia, Brazil) was founded in 1557.
  Fact 0.191: Merida (Merida, Venezuela) was founded in 1558.
  Fact 0.192: Londres (Catamarca, Argentina) was founded in 1558.
  Fact 0.193: Londres is the second oldest continuously inhabited European established settlement in Argentina.
  Fact 0.194: Londres was refounded and relocated several times.
  Fact 0.195: Mogi das Cruzes (Sao Paulo, Brazil) was founded in 1560.
  Fact 0.196: Ipojuca (Pernambuco, Brazil) was founded in 1560.
  Fact 0.197: Santa Cruz de la Sierra (Santa Cruz, Bolivia) was founded in 1561.
  Fact 0.198: Mendoza (Mendoza, Argentina) was founded in 1561.
  Fact 0.199: San Cristobal (Tachira, Venezuela) was founded in 1561.
  Fact 0.200: Itanhaem (Sao Paulo, Brasil) was founded in 1561.
  Fact 0.201: San Juan (San Juan, Argentina) was founded in 1562.
  Fact 0.202: Rio Grande (Zacatecas, Mexico) was founded in 1562.
  Fact 0.203: Susticacan (Zacatecas, Mexico) was founded in 1562.
  Fact 0.204: Cartago (Cartago, Costa Rica) was founded in 1563.
  Fact 0.205: Cartago is the oldest continuously inhabited European established settlement in Costa Rica.
  Fact 0.206: Sao Mateus (Espirito Santo, Brazil) was founded in 1564.
  Fact 0.207: Rio de Janeiro (Rio de Janeiro, Brazil) was founded in 1565.
  Fact 0.208: Saint Augustine (Florida, United States) was founded in 1565.
  Fact 0.209: Saint Augustine is the oldest continuously inhabited European established settlement in the continental United States.
  Fact 0.210: Saint Augustine was preceded by Puerto Rican colonies, Pensacola (Florida), which was destroyed in 1559, and Fort Caroline, which was destroyed in 1565.
  Fact 0.211: Tucuman (Tucuman, Argentina) was founded in 1565.
  Fact 0.212: Caracas (Capital District, Venezuela) was founded in 1567.
  Fact 0.213: Mazapil (Zacatecas, Mexico) was founded in 1568.
  Fact 0.214: Goiana (Pernambuco, Brazil) was founded in 1568.
  Fact 0.215: Goiana was declared a vila in 1685 and a cidade in 1840.
  Fact 0.216: Valaparaiso (Zacatecas, Mexico) was founded in 1568.
  Fact 0.217: Santa Ana (Santa Ana, El Salvador) was founded in 1569.
  Fact 0.218: Santa Ana was built over the Mayan city of Sihuatehuacan.
  Fact 0.219: Jerez (Zacatecas, Mexico) was founded in 1570.
  Fact 0.220: Sombrerete (Zacatecas, Mexico) was founded in 1570.
  Fact 0.221: Cochabamba (Cochabamba, Bolivia) was founded in 1571.
  Fact 0.222: Huancavelica (Huancavelica, Peru) was founded in 1572.
  Fact 0.223: San German (Puerto Rico, United States) was founded in 1573.
  Fact 0.224: San German is the second oldest European settlement in Puerto Rico.
  Fact 0.225: Cordoba (Cordoba, Argentina) was founded in 1573.
  Fact 0.226: Santa Fe (Santa Fe, Argentina) was founded in 1573.
  Fact 0.227: Buga (Valle del Cauca, Colombia) was founded in 1573.
  Fact 0.228: Huaraz (Ancash, Peru) was founded in 1574.
  Fact 0.229: Tarija (Tarija Department, Bolivia) was founded in 1574.
  Fact 0.230: Aguascalientes (Aguascalientes, Mexico) was founded in 1575.
  Fact 0.231: Leon (Guanajuato, Mexico) was founded in 1576.
  Fact 0.232: Juayua (Sonsonate, El Salvador) was founded in 1577.
  Fact 0.233: Fort-Liberte (Nord-Est, Haiti) was founded in 1578.
  Fact 0.234: Fort-Liberte was originally called Bayaha by the Spanish.
  Fact 0.235: Fort-Liberte is the oldest continuously inhabited European established settlement in Haiti.
  Fact 0.236: Tegucigalpa (Francisco Morazan, Honduras) was founded in 1578.
  Fact 0.237: Coamo (Puerto Rico, United States) was founded in 1579.
  Fact 0.238: Coamo is the third oldest European settlement in Puerto Rico.
  Fact 0.239: Coamo is in the same area where the Tainos had their village of Guayama.
  Fact 0.240: Coamo became officially a town in 1616.
  Fact 0.241: Coamo was given the title of villa by Spanish Royal Decree in 1778.
  Fact 0.242: Buenos Aires (Argentina) was founded in 1580.
  Fact 0.243: Santana de Parnaiba (Sao Paulo, Brasil) was founded in 1580.
  Fact 0.244: Santo Domingo (Xenacoj, Sacatepequez, Guatemala) was founded in 1580.
  Fact 0.245: Salta (Salta, Argentina) was founded in 1582.
  Fact 0.246: St John's (Newfoundland and Labrador, Canada) was founded in 1583.
  Fact 0.247: St John's is the oldest continuously inhabited British settlement in the Americas.
  Fact 0.248: Roanoke (North Carolina, United States) was founded in 1584.
  Fact 0.249: Joao Pessoa (Paraiba, Brazil) was founded in 1585.
  Fact 0.250: Concepcion del Oro (Zacatecas, Mexico) was founded in 1587.
  Fact 0.251: Corrientes (Corrientes, Argentina) was founded in 1588.
  Fact 0.252: Sao Cristovao (Sergipe, Brazil) was founded in 1590.
  Fact 0.253: Sao Cristovao was the first capital of Sergipe.
  Fact 0.254: Sao Cristovao was formerly known as Sergipe d'El Rei.
  Fact 0.255: Jimenez del Teul (Zacatecas, Mexico) was founded in 1591.
  Fact 0.256: Juan Aldama (Zacatecas, Mexico) was founded in 1591.
  Fact 0.257: Paudalho (Pernambuco, Brazil) was founded in 1591.
  Fact 0.258: La Rioja (La Rioja, Argentina) was founded in 1591.
  Fact 0.259: San Salvador de Jujuy (Jujuy, Argentina) was founded in 1593.
  Fact 0.260: Pinos (Zacatecas, Mexico) was founded in 1594.
  Fact 0.261: San Luis (San Luis, Argentina) was founded in 1594.
  Fact 0.262: Monterrey (Nuevo Leon, Mexico) was founded in 1596.
  Fact 0.263: Villahermosa (Tabasco, Mexico) was founded in 1596.
  Fact 0.264: Villahermosa was originally called Villa Felipe II.
  Fact 0.265: Portobelo (Colon, Panama) was founded in 1597.
  Fact 0.266: Parras (Coahuila, Mexico) was founded in 1598.
  Fact 0.267: Natal (Rio Grande do Norte, Brazil) was founded in 1599.
  Fact 0.268: Tadoussac (Quebec, Canada) was founded in 1599.
  Fact 0.269: Tadoussac is the oldest continuously inhabited French established settlement in the Americas.
  Fact 0.270: Tadoussac is the oldest European established settlement in Quebec.
  Fact 0.271: Facatativa (Cundinamarca, Colombia) was founded in 1600.
  Fact 0.272: Zipaquira (Cundinamarca, Colombia) was founded in 1600.
  Fact 0.273: David (Chiriqui Province, Panama) was founded in 1602.
  Fact 0.274: Salamanca (Guanajuato, Mexico) was founded in 1603.
  Fact 0.275: Port Royal (Nova Scotia, Canada) was founded in 1604.
  Fact 0.276: Port Royal is the oldest continuously inhabited European established settlement in Nova Scotia.
  Fact 0.277: Bayaguana (Monte Plata, Dominican Republic) was founded in 1606.
  Fact 0.278: Ibarra (Imbabura, Ecuador) was founded in 1606.
  Fact 0.279: Oruro (Oruro, Bolivia) was founded in 1606.
  Fact 0.280: Jamestown (Virginia, United States) was founded in 1607.
  Fact 0.281: Jamestown was the first permanent English established settlement in the Americas.
  Fact 0.282: Quebec City (Quebec, Canada) was founded in 1608.
  Fact 0.283: Fortaleza (Ceara, Brazil) was founded in 1609.
  Fact 0.284: Kecoughtan (Virginia, United States) was founded in 1610.
  Fact 0.285: Leon (Leon, Nicaragua) was originally founded in 1524 on the south-western shore of Lago Managua.
  Fact 0.286: Original Leon was destroyed by earthquakes.
  Fact 0.287: Leon was resettled 20 miles west in 1610 to its present-day location.
  Fact 0.288: Leon's original location is known as the ruins of Leon Viejo.
  Fact 0.289: Santa Fe (New Mexico, United States) was founded in 1610.
  Fact 0.290: Santa Fe is the oldest continuously inhabited state or territorial capital in the continental United States.
  Fact 0.291: Harbour Grace (Newfoundland and Labrador, Canada) was founded in 1610.
  Fact 0.292: Marechal Deodoro (Alagoas, Brazil) was founded in 1611.
  Fact 0.293: Sao Luis (Maranhao, Brazil) was founded in 1612.
  Fact 0.294: Sao Luis is the capital of Maranhao.
  Fact 0.295: St. George's (Bermuda) was founded in 1612.
  Fact 0.296: St. George's is the oldest continuously inhabited English established settlement in the Americas.
  Fact 0.297: Neiva (Huila, Colombia) was founded in 1612.
  Fact 0.298: Penedo (Alagoas, Brazil) was founded in 1613.
  Fact 0.299: Hopewell (Virginia, United States) was founded in 1613.
  Fact 0.300: Hopewell is the oldest continuously inhabited English settlement in North America.
  Fact 0.301: Hopewell was founded in 1613 as Bermuda City by Thomas Dale.
  Fact 0.302: Albany (New York, United States) was founded in 1614.
  Fact 0.303: Albany is the oldest US settlement north of Virginia.
  Fact 0.304: Albany is the second oldest state or territorial capital in the continental United States.
  Fact 0.305: Albany was incorporated in 1686.
  Fact 0.306: Sirinhaem (Pernambuco, Brazil) was founded in 1614.
  Fact 0.307: Sirinhaem was declared a vila in 1627 and a cidade in 1892.
  Fact 0.308: Cabo Frio (Rio de Janeiro, Brazil) was founded in 1615.
  Fact 0.309: Taos (New Mexico, United States) was founded in 1615.
  Fact 0.310: Ipiales (Narino, Colombia) was founded in 1615.
  Fact 0.311: Belem (Para, Brazil) was founded in 1616.
  Fact 0.312: Belem is the capital of Para.
  Fact 0.313: Medellin (Antioquia, Colombia) was founded in 1616 by Francisco de Herrera y Campuzano.
  Fact 0.314: Arecibo (Puerto Rico, United States) was founded in 1616.
  Fact 0.315: Jersey City (New Jersey, United States) was founded in 1617 as Pavonia, New Netherland.
  Fact 0.316: Cabo de Santo Agostinho (Pernambuco, Brazil) was founded in 1618.
  Fact 0.317: Cabo de Santo Agostinho was declared a vila in 1811 and a cidade in 1877.
  Fact 0.318: Ojocaliente (Zacatecas, Mexico) was founded in 1620.
  Fact 0.319: Plymouth (Massachusetts, United States) was founded in 1620.
  Fact 0.320: Bucaramanga (Santander, Colombia) was founded in 1622.
  Fact 0.321: Dover (New Hampshire, United States) was founded in 1623.
  Fact 0.322: Gloucester (Massachusetts, United States) was founded in 1623.
  Fact 0.323: Old Road (Saint Christopher, Saint Kitts and Nevis) was founded in 1624.
  Fact 0.324: Old Road is the oldest continuously inhabited British established settlement in the Caribbean.
  Fact 0.325: Old Road was the first successful non-Spanish established settlement in the Caribbean.
  Fact 0.326: Villa Soriano (Soriano, Uruguay) was founded in 1624.
  Fact 0.327: Villa Soriano is the oldest continuously inhabited European established settlement in Uruguay.
  Fact 0.328: New Amsterdam (New York, United States) was founded in 1625.
  Fact 0.329: New Amsterdam is now New York City.
  Fact 0.330: Quincy (Massachusetts, United States) was founded in 1625.
  Fact 0.331: Salem (Massachusetts, United States) was founded in 1626.
  Fact 0.332: Vitoria de Santo Antao (Pernambuco, Brazil) was founded in 1626.
  Fact 0.333: Vitoria de Santo Antao was declared a vila in 1811 and a cidade in 1843.
  Fact 0.334: Basseterre (Saint Christopher, Saint Kitts and Nevis) was founded in 1627.
  Fact 0.335: Basseterre is the oldest continuously inhabited French established settlement in the Caribbean.
  Fact 0.336: Bridgetown (Saint Michael, Barbados) was founded in 1628.
  Fact 0.337: Bridgetown is the second oldest continuously inhabited English established settlement in the Caribbean.
  Fact 0.338: Boston (Massachusetts, United States) was founded in 1630.
  Fact 0.339: Paramaribo (Paramaribo, Suriname) was founded in 1630.
  Fact 0.340: Paramaribo is the oldest continuously inhabited European established settlement in Suriname.
  Fact 0.341: Speightstown (Saint Peter, Barbados) was founded in 1630.
  Fact 0.342: Biddeford (Maine, United States) was founded in 1631.
  Fact 0.343: Hidalgo del Parral (Chihuahua, Mexico) was founded in 1631.
  Fact 0.344: Saint John (New Brunswick, Canada) was founded in 1631.
  Fact 0.345: Batopilas (Chihuahua, Mexico) was founded in 1632.
  Fact 0.346: Williamsburg (Virginia, United States) was founded in 1632.
  Fact 0.347: St. John's (Antigua and Barbuda) was founded in 1632.
  Fact 0.348: Green Bay (Wisconsin, United States) was founded in 1634.
  Fact 0.349: Green Bay is the oldest continuously inhabited French established settlement in the United States.
  Fact 0.350: St. Mary's City (Maryland, United States) was founded in 1634.
  Fact 0.351: Trois-Rivieres (Quebec, Canada) was founded in 1634.
  Fact 0.352: Willemstad (Curacao, Netherlands) was founded in 1634.
  Fact 0.353: Willemstad is the oldest continuously inhabited Dutch established settlement in the Caribbean.
  Fact 0.354: Concord (Massachusetts, United States) was founded in 1635.
  Fact 0.355: Old Saybrook (Connecticut, United States) was founded in 1635.
  Fact 0.356: Old Saybrook was originally a Dutch settlement called Kievits Hoek.
  Fact 0.357: Springfield (Massachusetts, United States) was founded in 1636.
  Fact 0.358: Providence (Rhode Island, United States) was founded in 1636.
  Fact 0.359: Hartford (Connecticut, United States) was founded in 1637.
  Fact 0.360: Belize City (Belize) was founded in 1638.
  Fact 0.361: Belize City is the oldest continuously inhabited European established settlement in Belize.
  Fact 0.362: Belize City was originally a Mayan city called Holzuz.
  Fact 0.363: Fort-de-France (Martinique) was founded in 1638.
  Fact 0.364: Fort-de-France was originally called Fort Saint Louis.
  Fact 0.365: New Haven (Connecticut, United States) was founded in 1638.
  Fact 0.366: Wilmington (Delaware, United States) was founded in 1638.
  Fact 0.367: Wilmington grew from Fort Christina, part of the New Sweden colony.
  Fact 0.368: Wilmington is the oldest continuously inhabited Swedish established settlement in the Americas.
  Fact 0.369: Cambridge (Massachusetts, United States) was founded in 1638.
  Fact 0.370: Rincon de Romos (Aguascalientes, Mexico) was founded in 1639.
  Fact 0.371: St. Marks (Florida, United States) was founded in 1639 or before.
  Fact 0.372: Barnstable (Massachusetts, United States) was founded in 1639.
  Fact 0.373: Barnstable is the second-oldest municipality in the United States and the only city on Cape Cod.
  Fact 0.374: Bridgeport (Connecticut, United States) was founded in 1639.
  Fact 0.375: Guilford (Connecticut, United States) was founded in 1639.
  Fact 0.376: Guilford is the location of the Henry Whitfield House.
  Fact 0.377: The Henry Whitfield House is the oldest house in Connecticut.
  Fact 0.378: The Henry Whitfield House is the oldest stone house in New England.
  Fact 0.379: Milford (Connecticut, United States) was founded in 1639.
  Fact 0.380: Newport (Rhode Island, United States) was founded in 1639.
  Fact 0.381: Tumaco (Narino, Colombia) was founded in 1640.
  Fact 0.382: Montreal (Quebec, Canada) was founded in 1642.
  Fact 0.383: Lexington (Massachusetts, United States) was founded in 1642.
  Fact 0.384: Warwick (Rhode Island, United States) was founded in 1642.
  Fact 0.385: Basse-Terre (Guadeloupe, France) was founded in 1643.
  Fact 0.386: Basse-Terre is the territorial capital.
  Fact 0.387: Dolores Hidalgo (Guanajuato, Mexico) was founded in 1643.
  Fact 0.388: Salvatierra (Guanajuato, Mexico) was founded in 1644.
  Fact 0.389: New London (Connecticut, United States) was founded in 1646.
  Fact 0.390: Alcantara (Maranhao, Brazil) was founded in 1648.
  Fact 0.391: Paranagua (Parana, Brazil) was founded in 1648.
  Fact 0.392: Annapolis (Maryland, United States) was founded in 1649.
  Fact 0.393: Castries (Castries, Saint Lucia) was founded in 1650.
  Fact 0.394: Saint George's (Saint George, Grenada) was founded in 1650.
  Fact 0.395: Norwalk (Connecticut, United States) was founded in 1651.
  Fact 0.396: Vieux Fort (Vieux Fort, Saint Lucia) was founded in 1654.
  Fact 0.397: Ciudad Juarez (Chihuahua, Mexico) was founded in 1659.
  Fact 0.398: El Paso (Texas, United States) was founded in 1659.
  Fact 0.399: Charlestown (Nevis, St Kitts & Nevis) was founded in 1660.
  Fact 0.400: Placentia (Newfoundland and Labrador, Canada) was founded in 1660.
  Fact 0.401: Placentia was the French capital until 1713.
  Fact 0.402: Placentia was originally called Plaisance.
  Fact 0.403: Rye (New York, United States) was founded in 1660.
  Fact 0.404: Cayenne (French Guiana, France) was founded in 1664.
  Fact 0.405: Cayenne is the oldest continuously inhabited European established settlement in French Guiana.
  Fact 0.406: Quibdo (Choco, Colombia) was founded in 1664.
  Fact 0.407: Port-de-Paix (Nord-Ouest, Haiti) was founded in 1665.
  Fact 0.408: Codrington (Barbuda, Antigua and Barbuda) was founded in 1666.
  Fact 0.409: Newark (New Jersey, United States) was founded in 1666.
  Fact 0.410: Paraty (Rio de Janeiro, Brazil) was founded in 1667.
  Fact 0.411: Sault Ste. Marie (Michigan, United States) was founded in 1668.
  Fact 0.412: Sault Ste. Marie is the oldest city in Michigan.
  Fact 0.413: Boqueirao (Paraiba, Brazil) was founded in 1670.
  Fact 0.414: Cap-Haitien (Nord, Haiti) was founded in 1670.
  Fact 0.415: Cap-Haitien was originally called Cap-Francais.
  Fact 0.416: Charleston (South Carolina, United States) was founded in 1670.
  Fact 0.417: Charlotte Amalie (St. Thomas, U.S. Virgin Islands, United States) was founded in 1672.
  Fact 0.418: Charlotte Amalie is the oldest permanent European settlement on Saint Thomas island.
  Fact 0.419: Charlotte Amalie is the oldest continuously inhabited Danish established settlement in the Americas.
  Fact 0.420: Kingston (Ontario, Canada) was founded in 1673.
  Fact 0.421: Kingston grew from Fort Frontenac.
  Fact 0.422: Kingston has been continuously inhabited since 1784.
  Fact 0.423: San Jose de Gracia (Aguascalientes, Mexico) was founded in 1673.
  Fact 0.424: Worcester (Massachusetts, United States) was founded in 1673.
  Fact 0.425: Cachoeira (Bahia, Brazil) was founded in 1674.
  Fact 0.426: Waterbury (Connecticut, United States) was founded in 1674.
  Fact 0.427: Crosswicks (New Jersey, United States) was founded in 1677.
  Fact 0.428: Crosswicks is a Quaker settlement.
  Fact 0.429: Crosswicks has been continuously inhabited since its founding.
  Fact 0.430: South Orange (New Jersey, United States) was founded in 1680.
  Fact 0.431: South Orange grew from Newark, later called Orange.
  Fact 0.432: Colonia del Sacramento (Colonia, Uruguay) was founded in 1680.
  Fact 0.433: Cockburn Town (Turks and Caicos Islands) was founded in 1681.
  Fact 0.434: Philadelphia (Pennsylvania, United States) was founded in 1681.
  Fact 0.435: Norfolk (Virginia, United States) was founded in 1682.
  Fact 0.436: Sao Borja (Rio Grande do Sul, Brazil) was founded in 1682.
  Fact 0.437: Dover (Delaware, United States) was founded in 1683.
  Fact 0.438: Escada (Pernambuco, Brazil) was founded in 1685.
  Fact 0.439: Trinidad (Beni Department, Bolivia) was founded in 1686.
  Fact 0.440: New Britain (Connecticut, United States) was founded in 1687.
  Fact 0.441: Sao Luiz Gonzaga (Rio Grande do Sul, Brazil) was founded in 1687.
  Fact 0.442: Sao Miguel das Missoes (Rio Grande do Sul, Brazil) was founded in 1687.
  Fact 0.443: Sao Nicolau (Rio Grande do Sul, Brazil) was founded in 1687.
  Fact 0.444: Marigot (Saint Martin) was founded in 1689.
  Fact 0.445: San Gil (Santander, Colombia) was founded in 1689.
  Fact 0.446: Port of Spain (Trinidad, Trinidad and Tobago) was founded in 1690.
  Fact 0.447: Sao Lourenco das Missoes (Rio Grande do Sul, Brazil) was founded in 1690.
  Fact 0.448: Santa Rosa de Viterbo (Boyaca, Colombia) was founded in 1690.
  Fact 0.449: Villanueva (Zacatecas, Mexico) was founded in 1692.
  Fact 0.450: Bom Jesus da Lapa (Bahia, Brazil) was founded in 1693.
  Fact 0.451: Burlington (New Jersey, United States) was founded in 1693.
  Fact 0.452: Curitiba (Parana, Brazil) was founded in 1693.
  Fact 0.453: Kingston (St Andrew, Jamaica) was founded in 1693.
  Fact 0.454: Nassau (New Providence, Bahamas) was founded in 1695.
  Fact 0.455: Saint-Marc (Artibonite, Haiti) was founded in 1695.
  Fact 0.456: Assu (Rio Grande do Norte, Brazil) was founded in 1696.
  Fact 0.457: Assu was declared a vila in 1766 and a cidade in 1845.
  Fact 0.458: Rimouski (Quebec, Canada) was founded in 1696.
  Fact 0.459: Ambato (Tungurahua, Ecuador) was founded in 1698.
  Fact 0.460: Pensacola (Florida, United States) was founded in 1698.
  Fact 0.461: Pensacola is known for being the first European settlement in North America.
  Fact 0.462: Biloxi (Mississippi, United States) was founded in 1699.
  Fact 0.463: Baton Rouge (Louisiana, United States) was founded in 1699.
  Fact 0.464: Barreiras (Bahia, Brazil) was founded in 1700.
  Fact 0.465: Detroit (Michigan, United States) was founded in 1701.
  Fact 0.466: Sao Joao del Rei (Minas Gerais, Brazil) was founded in 1701.
  Fact 0.467: Jesus Maria (Aguascalientes, Mexico) was founded in 1702.
  Fact 0.468: Mobile (Alabama, United States) was founded in 1702.
  Fact 0.469: Kaskaskia (Illinois, United States) was founded in 1703.
  Fact 0.470: Bath (North Carolina, United States) was founded in 1705.
  Fact 0.471: Albuquerque (New Mexico, United States) was founded in 1706.
  Fact 0.472: Santo Angelo (Rio Grande do Sul, Brazil) was founded in 1706.
  Fact 0.473: Chihuahua (Chihuahua, Mexico) was founded in 1709.
  Fact 0.474: Chatham (New Jersey, United States) was founded in 1710.
  Fact 0.475: Chatham was founded on land purchased in 1680.
  Fact 0.476: New Bern (North Carolina, United States) was founded in 1710.
  Fact 0.477: New Bern is the birthplace of Pepsi.
  Fact 0.478: Beaufort (South Carolina, United States) was founded in 1711.
  Fact 0.479: Ouro Preto (Minas Gerais, Brazil) was founded in 1711.
  Fact 0.480: Natchitoches (Louisiana, United States) was founded in 1714.
  Fact 0.481: Natchitoches is the oldest settlement in the Louisiana Purchase.
  Fact 0.482: Antonina (Parana, Brazil) was founded in 1714.
  Fact 0.483: Laguna (Santa Catarina, Brazil) was founded in 1714.
  Fact 0.484: Serro (Minas Gerais, Brazil) was founded in 1714.
  Fact 0.485: Kekionga (Indiana, United States) was founded in 1715 or before.
  Fact 0.486: Kekionga is the capital of the Miami tribe.
  Fact 0.487: Natchez (Mississippi, United States) was founded in 1716.
  Fact 0.488: Natchez dates to the founding of Fort Rosalie by the French.
  Fact 0.489: Brejo do Cruz (Paraiba, Brazil) was founded in 1717.
  Fact 0.490: Zacatecas (Zacatecas, Mexico) was founded in 1717.
  Fact 0.491: Westborough (Massachusetts, United States) was founded in 1717.
  Fact 0.492: Westborough was the one hundredth town in Massachusetts.
  Fact 0.493: New Orleans (Louisiana, United States) was founded in 1718.
  Fact 0.494: Rio de Contas (Bahia, Brazil) was founded in 1718.
  Fact 0.495: San Antonio (Texas, United States) was founded in 1718.
  Fact 0.496: Tiradentes (Minas Gerais, Brazil) was founded in 1718.
  Fact 0.497: Trenton (New Jersey, United States) was founded in 1719.
  Fact 0.498: Cortazar (Guanajuato, Mexico) was founded in 1721.
  Fact 0.499: Edenton (North Carolina, United States) was founded in 1722.
  Fact 0.500: Guaynabo (Puerto Rico, United States) was founded in 1723.
  Fact 0.501: Guaynabo was declared a pueblo in 1723 and a municipio in 1768.
  Fact 0.502: Brattleboro (Vermont, United States) was founded in 1724.
  Fact 0.503: Brattleboro grew out of Fort Dummer.
  Fact 0.504: Concord (New Hampshire, United States) was founded in 1725.
  Fact 0.505: Florianopolis (Santa Catarina, Brazil) was founded in 1726.
  Fact 0.506: Montevideo (Montevideo, Uruguay) was founded in 1726.
  Fact 0.507: Cuiaba (Mato Grosso, Brazil) was founded in 1727.
  Fact 0.508: Goias (Goias, Brazil) was founded in 1727.
  Fact 0.509: Pirenopolis (Goias, Brazil) was founded in 1727.
  Fact 0.510: Fredericksburg (Virginia, United States) was founded in 1728.
  Fact 0.511: Inhambupe (Bahia, Brazil) was founded in 1728.
  Fact 0.512: Nuuk (Greenland) was founded in 1728.
  Fact 0.513: Nuuk was originally a Norse colony called Godthab.
  Fact 0.514: Nuuk is the oldest continuously inhabited European established settlement in Greenland.
  Fact 0.515: Baltimore (Maryland, United States) was founded in 1729.
  Fact 0.516: Roseau (Saint George, Dominica) was founded in 1730.
  Fact 0.517: Fredericton (New Brunswick, Canada) was founded in 1732.
  Fact 0.518: Fredericton was originally called Ste. Anne's Point.
  Fact 0.519: Vincennes (Indiana, United States) was founded in 1732.
  Fact 0.520: Morretes (Parana, Brazil) was founded in 1733.
  Fact 0.521: Pau dos Ferros (Rio Grande do Norte, Brazil) was founded in 1733.
  Fact 0.522: Richmond (Virginia, United States) was founded in 1733.
  Fact 0.523: Savannah (Georgia, United States) was founded in 1733.
  Fact 0.524: Cucuta (Norte de Santander, Colombia) was founded in 1733.
  Fact 0.525: Augusta (Georgia, United States) was founded in 1736.
  Fact 0.526: Rio Grande (Rio Grande do Sul, Brazil) was founded in 1737.
  Fact 0.527: Fort Rouge (Manitoba, Canada) was founded in 1738.
  Fact 0.528: Fort Rouge is now Winnipeg, Manitoba.
  Fact 0.529: San Jose (San Jose, Costa Rica) was founded in 1738.
  Fact 0.530: Santa Rosalia de Camargo (Chihuahua, Mexico) was founded in 1740.
  Fact 0.531: Portalegre (Rio Grande do Norte, Brazil) was founded in 1740.
  Fact 0.532: San Felipe de Puerto Plata (Puerto Plata, Dominican Republic) was founded in 1740.
  Fact 0.533: Viamao (Rio Grande do Sul, Brazil) was founded in 1741.
  Fact 0.534: Canguaretama (Rio Grande do Norte, Brazil) was founded in 1743.
  Fact 0.535: Canguaretama was declared a vila in 1858 and a cidade in 1885.
  Fact 0.536: Caico (Rio Grande do Norte, Brazil) was founded in 1748.
  Fact 0.537: Caico was declared a vila in 1788 and a cidade in 1868.
  Fact 0.538: Alexandria (Virginia, United States) was founded in 1749.
  Fact 0.539: San Ignacio de Velasco (Santa Cruz Department, Bolivia) was founded in 1749.
  Fact 0.540: Port-au-Prince (Ouest, Haiti) was founded in 1749.
  Fact 0.541: Halifax (Nova Scotia, Canada) was founded in 1749.
  Fact 0.542: Windsor (Ontario, Canada) was founded in 1749.
  Fact 0.543: Windsor is the oldest continually inhabited settlement in Canada west of Montreal.
  Fact 0.544: Manchester (New Hampshire, United States) was founded in 1751.
  Fact 0.545: Manchester is a former American textile capital.
  Fact 0.546: Manchester is named after Manchester, England.
  Fact 0.547: Buique (Pernambuco, Brazil) was founded in 1752.
  Fact 0.548: Patos (Paraiba, Brazil) was founded in 1752.
  Fact 0.549: Patos was declared a vila in 1832 and a cidade in 1903.
  Fact 0.550: Tubac (Arizona, United States) was founded in 1752.
  Fact 0.551: Augusta (Maine, United States) was founded in 1754.
  Fact 0.552: Charlotte (North Carolina, United States) was founded in 1755.
  Fact 0.553: Salto (Salto, Uruguay) was founded in 1756.
  Fact 0.554: Santa Barbara de Samana (Samana, Dominican Republic) was founded in 1756.
  Fact 0.555: Yauco (Puerto Rico, United States) was founded in 1756.
  Fact 0.556: Estancia (Sergipe, Brazil) was founded in 1757.
  Fact 0.557: Pittsburgh (Pennsylvania, United States) was founded in 1758.
  Fact 0.558: Macapa (Amapa, Brazil) was founded in 1758.
  Fact 0.559: Ares (Rio Grande do Norte, Brazil) was founded in 1760.
  Fact 0.560: Sabana de la Mar (Hato Mayor, Dominican Republic) was founded in 1760.
  Fact 0.561: Sacaba (Cochabamba Department, Bolivia) was founded in 1761.
  Fact 0.562: Kingstown (St Vincent, Saint Vincent and the Grenadines) was founded in 1762.
  Fact 0.563: Parnaiba (Piaui, Brazil) was founded in 1762.
  Fact 0.564: Shepherdstown (West Virginia, United States) was founded in 1762.
  Fact 0.565: Shepherdstown was originally known as Mecklenburg.
  Fact 0.566: Allentown (Pennsylvania, United States) was founded in 1762.
  Fact 0.567: Allentown was incorporated as Northamptontown.
  Fact 0.568: Baia da Traicao (Paraiba, Brazil) was founded in 1762.
  Fact 0.569: Burlington (Vermont, United States) was founded in 1763.
  Fact 0.570: Panfilo Natera (Zacatecas, Mexico) was founded in 1763.
  Fact 0.571: Ellsworth (Maine, United States) was founded in 1763.
  Fact 0.572: Philipsburg (Sint Maarten, Netherlands) was founded in 1763.
  Fact 0.573: St. Louis (Missouri, United States) was founded in 1763.
  Fact 0.574: Charlottetown (Prince Edward Island, Canada) was founded in 1764.
  Fact 0.575: Pointe-a-Pitre (Grand Terre, Guadeloupe) was founded in 1764.
  Fact 0.576: Port Elizabeth (Grenadines, Saint Vincent and the Grenadines) was founded in 1765.
  Fact 0.577: Portsmouth (Saint John, Dominica) was founded in 1765.
  Fact 0.578: Ceara-Mirim (Rio Grande do Norte, Brazil) was founded in 1767.
  Fact 0.579: Araioses (Maranhao, Brazil) was founded in 1767.
  Fact 0.580: Elizabethton (Tennessee, United States) was founded in 1769.
  Fact 0.581: Falmouth (Trelawny, Jamaica) was founded in 1769.
  Fact 0.582: San Diego (California, United States) was founded in 1769.
  Fact 0.583: San Diego grew from the Presidio of San Diego.
  Fact 0.584: Lapa (Parana, Brazil) was founded in 1769.
  Fact 0.585: Monterey (California, United States) was founded in 1770.
  Fact 0.586: Monterey grew from the Presidio of Monterey.
  Fact 0.587: Monterey was the original capital of California.
  Fact 0.588: Araxa (Minas Gerais, Brazil) was founded in 1770.
  Fact 0.589: Calvillo (Aguascalientes, Mexico) was founded in 1771.
  Fact 0.590: Porto Alegre (Rio Grande do Sul, Brazil) was founded in 1772.
  Fact 0.591: Tucson (Arizona, United States) was founded in 1775.
  Fact 0.592: Castro (Parana, Brazil) was founded in 1774.
  Fact 0.593: Lexington (Kentucky, United States) was founded in 1775.
  Fact 0.594: Boonesborough (Kentucky, United States) was founded in 1775.
  Fact 0.595: Boonesborough grew from Fort Boonesborough, built by pioneer Daniel Boone.
  Fact 0.596: Sao Lourenco da Mata (Pernambuco, Brazil) was founded in 1775.
  Fact 0.597: Sao Lourenco da Mata had district status in 1775, village status in 1884, and municipality status in 1890.
  Fact 0.598: San Francisco (California, United States) was founded in 1776.
  Fact 0.599: Sincelejo (Sucre, Colombia) was founded in 1776.
  Fact 0.600: San Jose (California, United States) was founded in 1777.
  Fact 0.601: San Jose was originally known as El Pueblo de San Jose de Guadalupe.
  Fact 0.602: San Jose was the first town in the Spanish colony of Nueva California, which later became Alta California.
  Fact 0.603: Monteria (Cordoba, Colombia) was founded in 1777.
  Fact 0.604: Corumba (Mato Grosso do Sul, Brazil) was founded in 1778.
  Fact 0.605: Louisville (Kentucky, United States) was founded in 1778.
  Fact 0.606: Louisville grew from Fort Nelson, established by explorer George Rogers Clark.
  Fact 0.607: San Francisco de Macoris (Duarte, Dominican Republic) was founded in 1778.
  Fact 0.608: Jonesborough (Tennessee, United States) was founded in 1779.
  Fact 0.609: Jonesborough was later organized as the lost State of Franklin, with Jonesborough as capital, in 1784.
  Fact 0.610: Nashville (Tennessee, United States) was founded in 1779.
  Fact 0.611: Nashville grew from Fort Nashborough.
  Fact 0.612: Codo (Maranhao, Brazil) was founded in 1780.
  Fact 0.613: Codo was declared a vila in 1833 and a cidade in 1896.
  Fact 0.614: Las Matas de Farfan (San Juan, Dominican Republic) was founded in 1780.
  Fact 0.615: Arauca (Arauca, Colombia) was founded in 1780.
  Fact 0.616: Montpelier (Vermont, United States) was founded in 1781.
  Fact 0.617: Los Angeles (California, United States) was founded in 1781.
  Fact 0.618: Catu (Bahia, Brazil) was founded in 1782.
  Fact 0.619: Georgetown (Demerara-Mahaica, Guyana) was founded in 1782.
  Fact 0.620: Georgetown was originally called La Nouvelle Ville.
  Fact 0.621: Vitoria da Conquista (Bahia, Brazil) was founded in 1783.
  Fact 0.622: San Fernando (Trinidad, Trinidad and Tobago) was founded in 1784.
  Fact 0.623: Harrisburg (Pennsylvania, United States) was founded in 1785.
  Fact 0.624: North Battleford (Saskatchewan, Canada) was founded in 1785 as a fur trading post.
  Fact 0.625: Battleford was the capital of the North West Territories in 1877.
  Fact 0.626: North Battleford was incorporated as a city with a population of 5000 or more in 1913.
  Fact 0.627: Asheville (North Carolina, United States) was founded in 1785.
  Fact 0.628: Gustavia (Saint Barthelemy) was founded in 1785.
  Fact 0.629: Huntington (West Virginia, United States) was founded in 1785.
  Fact 0.630: Sydney (Nova Scotia, Canada) was founded in 1785.
  Fact 0.631: Columbia (South Carolina, United States) was founded in 1786.
  Fact 0.632: Escada (Pernambuco, Brazil) was declared a vila in 1854 and a cidade in 1873.
  Fact 0.633: Florissant (Missouri, United States) was founded in 1786.
  Fact 0.634: Florissant was originally known as St. Ferdinand.
  Fact 0.635: Frankfort (Kentucky, United States) was founded in 1786.
  Fact 0.636: Portland (Maine, United States) was founded in 1786.
  Fact 0.637: Boa Ventura (Paraiba, Brazil) was founded in 1787.
  Fact 0.638: Marietta (Ohio, United States) was founded in 1788.
  Fact 0.639: Cincinnati (Ohio, United States) was founded in 1788.
  Fact 0.640: Charleston (West Virginia, United States) was founded in 1788.
  Fact 0.641: Charleston grew from Fort Lee.
  Fact 0.642: Mercedes (Soriano, Uruguay) was founded in 1788.
  Fact 0.643: Buffalo (New York, United States) was founded in 1789.
  Fact 0.644: Itambe (Pernambuco, Brazil) was founded in 1789.
  Fact 0.645: Itambe was declared a vila in 1867 and a cidade in 1879.
  Fact 0.646: Hamilton (Bermuda) was founded in 1790.
  Fact 0.647: Washington (District of Columbia, United States) was founded in 1790.
  Fact 0.648: Jacksonville (Florida, United States) was founded in 1791.
  Fact 0.649: Jacksonville was known as Cowford.
  Fact 0.650: Knoxville (Tennessee, United States) was founded in 1791.
  Fact 0.651: Bangor (Maine, United States) was founded in 1791.
  Fact 0.652: Raleigh (North Carolina, United States) was founded in 1792.
  Fact 0.653: Sherbrooke (Quebec, Canada) was founded in 1793.
  Fact 0.654: Toronto (Ontario, Canada) was founded in 1793.
  Fact 0.655: Fort Wayne (Indiana, United States) was founded in 1794.
  Fact 0.656: Edmonton (Alberta, Canada) was founded in 1795.
  Fact 0.657: Edmonton grew from Fort Edmonton.
  Fact 0.658: Erie (Pennsylvania, United States) was founded in 1795.
  Fact 0.659: Erie grew from the French Fort Presque Isle.
  Fact 0.660: Maryville (Tennessee, United States) was founded in 1795.
  Fact 0.661: Maryville grew from the American Fort Craig.
  Fact 0.662: Cleveland (Ohio, United States) was founded in 1796.
  Fact 0.663: Oranjestad (Aruba, Netherlands) was founded in 1796.
  Fact 0.664: Chaguanas (Trinidad, Trinidad and Tobago) was founded in 1797.
  Fact 0.665: Franklinton (Ohio, United States) was founded in 1797.
  Fact 0.666: Franklinton was eventually absorbed by Columbus, Ohio.
  Fact 0.667: Araruama (Rio de Janeiro, Brazil) was founded in 1799.
  Fact 0.668: Hull (Quebec, Canada) was founded in 1800.
  Fact 0.669: Hull was formerly known as Wrightville.
  Fact 0.670: Sonson (Antioquia, Colombia) was founded in 1800.
  Fact 0.671: Chapadinha (Maranhao, Brazil) was founded in 1802.
  Fact 0.672: Santa Cruz de Barahona (Barahona, Dominican Republic) was founded in 1802.
  Fact 0.673: Chicago (Illinois, United States) was founded in 1803.
  Fact 0.674: Chicago grew from Fort Dearborn.
  Fact 0.675: Huntsville (Alabama, United States) was founded in 1805.
  Fact 0.676: Hot Springs (Arkansas, United States) was founded in 1807.
  Fact 0.677: Agua Preta (Pernambuco, Brazil) was founded in 1809.
  Fact 0.678: Agua Preta was declared a vila in 1846 and a cidade in 1895.
  Fact 0.679: Guarapuava (Parana, Brazil) was founded in 1810.
  Fact 0.680: Manchester (New Hampshire, United States) was founded in 1810.
  Fact 0.681: Pilao Arcado (Bahia, Brazil) was founded in 1810.
  Fact 0.682: San Bernardino (California, United States) was founded in 1810.
  Fact 0.683: Garanhuns (Pernambuco, Brazil) was founded in 1811.
  Fact 0.684: Astoria (Oregon, United States) was founded in 1811.
  Fact 0.685: Columbus (Ohio, United States) was founded in 1812.
  Fact 0.686: Pelotas (Rio Grande do Sul, Brazil) was founded in 1812.
  Fact 0.687: Pelotas was declared a vila in 1832 and a cidade in 1835.
  Fact 0.688: Capela (Sergipe, Brazil) was founded in 1813.
  Fact 0.689: Chisec (Alta Verapaz, Guatemala) was founded in 1813.
  Fact 0.690: Barranquilla (Atlantico, Colombia) was founded in 1813.
  Fact 0.691: Areia (Paraiba, Brazil) was founded in 1815.
  Fact 0.692: Hamilton (Ontario, Canada) was founded in 1815.
  Fact 0.693: Alagoinhas (Bahia, Brazil) was founded in 1816.
  Fact 0.694: Chattanooga (Tennessee, United States) was founded in 1816.
  Fact 0.695: Chattanooga was originally named Ross's Landing.
  Fact 0.696: Saginaw (Michigan, United States) was founded in 1816.
  Fact 0.697: Savaneta (Aruba, Netherlands) was founded in 1816.
  Fact 0.698: Fort Smith (Arkansas, United States) was founded in 1817.
  Fact 0.699: Itaguai (Rio de Janeiro, Brazil) was founded in 1818.
  Fact 0.700: Montgomery (Alabama, United States) was founded in 1819.
  Fact 0.701: Montgomery is near the site of the earlier French Fort Toulouse.
  Fact 0.702: Springfield (Illinois, United States) was founded in 1819.
  Fact 0.703: Managua (Managua, Nicaragua) was founded in 1819.
  Fact 0.704: Memphis (Tennessee, United States) was founded in 1819.
  Fact 0.705: Memphis is near the site of the earlier French Fort Prudhomme.
  Fact 0.706: Palmeira (Parana, Brazil) was founded in 1819.
  Fact 0.707: Tuscaloosa (Alabama, United States) was founded in 1819.
  Fact 0.708: Tuscaloosa is a former state capital.
  Fact 0.709: Tuscaloosa grew from former Native American settlements beginning in 1580.
  Fact 0.710: Cotegipe (Bahia, Brazil) was founded in 1820.
  Fact 0.711: Little Rock (Arkansas, United States) was founded in 1821.
  Fact 0.712: Indianapolis (Indiana, United States) was founded in 1821.
  Fact 0.713: Jefferson City (Missouri, United States) was founded in 1821.
  Fact 0.714: Decatur (Alabama, United States) was founded in 1821.
  Fact 0.715: Ponta Grossa (Parana, Brazil) was founded in 1822.
  Fact 0.716: Jackson (Mississippi, United States) was founded in 1822.
  Fact 0.717: Tampa (Florida, United States) was founded in 1823.
  Fact 0.718: Tampa grew from the earlier military post Fort Brooke.
  Fact 0.719: Vancouver (Washington, United States) was founded in 1824.
  Fact 0.720: Tallahassee (Florida, United States) was founded in 1824.
  Fact 0.721: Vicksburg (Mississippi, United States) was founded in 1825.
  Fact 0.722: Grand Rapids (Michigan, United States) was founded in 1825.
  Fact 0.723: Syracuse (New York, United States) was founded in 1825 as a village.
  Fact 0.724: Syracuse received a city charter in 1847.
  Fact 0.725: Ipiranga (Parana, Brazil) was founded in 1826.
  Fact 0.726: London (Ontario, Canada) was founded in 1826.
  Fact 0.727: Bytown (Ontario, Canada) was founded in 1826.
  Fact 0.728: Bytown is now Ottawa, Ontario.
  Fact 0.729: Fort Leavenworth (Kansas, United States) was founded in 1827.
  Fact 0.730: St. Andrews (Florida, United States) was founded in 1827.
  Fact 0.731: St. Andrews is now part of Panama City.
  Fact 0.732: Key West (Florida, United States) was founded in 1828.
  Fact 0.733: Greenville (South Carolina, United States) was founded in 1831.
  Fact 0.734: Piracanjuba (Goias, Brazil) was founded in 1831.
  Fact 0.735: Goianinha (Rio Grande do Norte, Brazil) was founded in 1832.
  Fact 0.736: Touros (Rio Grande do Norte, Brazil) was founded in 1832.
  Fact 0.737: Bananeiras (Paraiba, Brazil) was founded in 1833.
  Fact 0.738: Juazeiro (Bahia, Brazil) was founded in 1833.
  Fact 0.739: Kitchener (Ontario, Canada) was founded in 1833.
  Fact 0.740: Kitchener was originally called Berlin.
  Fact 0.741: Milwaukee (Wisconsin, United States) was founded in 1833.
  Fact 0.742: Nazare da Mata (Pernambuco, Brazil) was founded in 1833.
  Fact 0.743: Nazare da Mata was declared a vila in 1833 and a cidade in 1850.
  Fact 0.744: Rio Formoso (Pernambuco, Brazil) was founded in 1833.
  Fact 0.745: Rio Formoso was declared a vila in 1833 and a cidade in 1850.
  Fact 0.746: Vassouras (Rio de Janeiro, Brazil) was founded in 1833.
  Fact 0.747: Austin (Texas, United States) was founded in 1835.
  Fact 0.748: Barra do Corda (Maranhao, Brazil) was founded in 1835.
  Fact 0.749: Curupuru (Maranhao, Brazil) was founded in 1835.
  Fact 0.750: Curupuru was declared a vila in 1841 and a cidade in 1920.
  Fact 0.751: Davenport (Iowa, United States) was founded in 1836.
  Fact 0.752: Oshawa (Ontario, Canada) was founded in 1836.
  Fact 0.753: Shreveport (Louisiana, United States) was founded in 1836.
  Fact 0.754: Madison (Wisconsin, United States) was founded in 1836.
  Fact 0.755: Tulsa (Oklahoma, United States) was founded in 1836.
  Fact 0.756: The Lochapoka named their new settlement Tulasi, meaning old town.
  Fact 0.757: Gloria do Goita (Pernambuco, Brazil) was founded in 1837.
  Fact 0.758: Gloria do Goita was declared a vila in 1877 and a cidade in 1884.
  Fact 0.759: Lansing (Michigan, United States) was founded in 1837.
  Fact 0.760: Houston (Texas, United States) was founded in 1837.
  Fact 0.761: Divina Pastora (Sergipe, Brazil) was founded in 1838.
  Fact 0.762: Kansas City (Missouri, United States) was founded in 1838.
  Fact 0.763: St Paul (Minnesota, United States) was founded in 1838.
  Fact 0.764: Santa Rita (Paraiba, Brazil) was founded in 1839.
  Fact 0.765: Santa Rita was declared a vila in 1890 and a cidade in 1924.
  Fact 0.766: Summerside (Prince Edward Island, Canada) was founded in 1840.
  Fact 0.767: Villavicencio (Meta, Colombia) was founded in 1840.
  Fact 0.768: Cedar Rapids (Iowa, United States) was founded in 1841.
  Fact 0.769: Dallas (Texas, United States) was founded in 1841.
  Fact 0.770: Piata (Bahia, Brazil) was founded in 1842.
  Fact 0.771: Salem (Oregon, United States) was founded in 1842.
  Fact 0.772: Atlanta (Georgia, United States) was founded in 1843.
  Fact 0.773: Atlanta was originally known as Terminus, later renamed Marthasville.
  Fact 0.774: Atlanta acquired the name Atlanta in 1847.
  Fact 0.775: Des Moines (Iowa, United States) was founded in 1843.
  Fact 0.776: Stanley (East Falkland, Falkland Islands) was founded in 1843.
  Fact 0.777: Uruguaiana (Rio Grande do Sul, Brazil) was founded in 1843.
  Fact 0.778: Uruguaiana was declared a vila in 1846 and a cidade in 1874.
  Fact 0.779: Victoria (British Columbia, Canada) was founded in 1843.
  Fact 0.780: Lencois (Bahia, Brazil) was founded in 1844.
  Fact 0.781: Araruna (Paraiba, Brazil) was founded in 1845.
  Fact 0.782: Portland (Oregon, United States) was founded in 1845.
  Fact 0.783: Alagoa Grande (Paraiba, Brazil) was founded in 1847.
  Fact 0.784: Alagoa Grande was declared a vila in 1865 and a cidade in 1908.
  Fact 0.785: Salt Lake City (Utah, United States) was founded in 1847.
  Fact 0.786: Salt Lake City was originally known as Great Salt Lake City.
  Fact 0.787: Caruaru (Pernambuco, Brazil) was founded in 1848.
  Fact 0.788: Caruaru was the first municipality in the Agreste region of Pernambuco state.
  Fact 0.789: Rio Verde (Goias, Brazil) was founded in 1848.
  Fact 0.790: Sacramento (California, United States) was founded in 1848.
  Fact 0.791: Provo (Utah, United States) was founded in 1849.
  Fact 0.792: Manizales (Caldas, Colombia) was founded in 1849.
  Fact 0.793: San Luis (Colorado, United States) was founded in 1851.
  Fact 0.794: Seattle (Washington, United States) was founded in 1851.
  Fact 0.795: Jerico (Antioquia, Colombia) was founded in 1851.
  Fact 0.796: Mossoro (Rio Grande do Norte, Brazil) was founded in 1852.
  Fact 0.797: Sao Jose dos Pinhais (Parana, Brazil) was founded in 1852.
  Fact 0.798: Colon (Colon, Panama) was founded in 1852.
  Fact 0.799: Olympia (Washington, United States) was founded in 1853.
  Fact 0.800: Omaha (Nebraska, United States) was founded in 1854.
  Fact 0.801: Anajatuba (Maranhao, Brazil) was founded in 1854.
  Fact 0.802: Topeka (Kansas, United States) was founded in 1854.
  Fact 0.803: Aracaju (Sergipe, Brazil) was founded in 1855.
  Fact 0.804: Lincoln (Nebraska, United States) was founded in 1856.
  Fact 0.805: Lincoln was originally called Lancaster.
  Fact 0.806: O'Fallon (Missouri, United States) was founded in 1856.
  Fact 0.807: Sioux Falls (South Dakota, United States) was founded in 1856.
  Fact 0.808: Boquim (Sergipe, Brazil) was founded in 1857.
  Fact 0.809: Carson City (Nevada, United States) was founded in 1858.
  Fact 0.810: Denver (Colorado, United States) was founded in 1858.
  Fact 0.811: Yankton (South Dakota, United States) was founded in 1859.
  Fact 0.812: Chico (California, United States) was founded in 1860.
  Fact 0.813: Boise (Idaho, United States) was founded in 1863.
  Fact 0.814: Wichita (Kansas, United States) was founded in 1863.
  Fact 0.815: Casper (Wyoming, United States) was founded in 1864.
  Fact 0.816: Casper was originally called Platte Bridge Station.
  Fact 0.817: Helena (Montana, United States) was founded in 1864.
  Fact 0.818: Salinas (California, United States) was founded in 1864.
  Fact 0.819: Prince Albert (Saskatchewan, Canada) was founded in 1866.
  Fact 0.820: Prince Albert was settled by Rev. James Nisbet.
  Fact 0.821: Prince Albert was incorporated as a city with a population of 5000 or more on October 8, 1904.
  Fact 0.822: Cheyenne (Wyoming, United States) was founded in 1867.
  Fact 0.823: Gameleira (Pernambuco, Brazil) was founded in 1867.
  Fact 0.824: Minneapolis (Minnesota, United States) was founded in 1867.
  Fact 0.825: Colinas (Maranhao, Brazil) was founded in 1868.
  Fact 0.826: Colinas was declared a vila in 1870 and a cidade in 1891.
  Fact 0.827: Phoenix (Arizona, United States) was founded in 1868.
  Fact 0.828: Afogados da Ingazeira (Pernambuco, Brazil) was founded in 1870.
  Fact 0.829: Campo Largo (Parana, Brazil) was founded in 1870.
  Fact 0.830: Puerto Limon (Limon, Costa Rica) was founded in 1870.
  Fact 0.831: Aguas Belas (Pernambuco, Brazil) was founded in 1871.
  Fact 0.832: Alto Parnaiba (Maranhao, Brazil) was founded in 1871.
  Fact 0.833: Birmingham (Alabama, United States) was founded in 1871.
  Fact 0.834: Colorado Springs (Colorado, United States) was founded in 1871.
  Fact 0.835: Fargo (North Dakota, United States) was founded in 1871.
  Fact 0.836: Imbituva (Parana, Brazil) was founded in 1871.
  Fact 0.837: Sao Joao do Triunfo (Parana, Brazil) was founded in 1871.
  Fact 0.838: Anniston (Alabama, United States) was founded in 1872.
  Fact 0.839: Bismarck (North Dakota, United States) was founded in 1872.
  Fact 0.840: Tibagi (Parana, Brazil) was founded in 1872.
  Fact 0.841: Jaboatao dos Guararapes (Pernambuco, Brazil) was founded in 1873.
  Fact 0.842: Jaboatao dos Guararapes had its first settlement in 1593, village status in 1873, and city status in 1884.
  Fact 0.843: Timbauba (Pernambuco, Brazil) was founded in 1873.
  Fact 0.844: Timbauba was declared a vila in 1879 and a cidade in 1884.
  Fact 0.845: Orlando (Florida, United States) was founded in 1875.
  Fact 0.846: Calgary (Alberta, Canada) was founded in 1875.
  Fact 0.847: Calgary grew from Fort Calgary.
  Fact 0.848: Humboldt (Saskatchewan, Canada) was founded in 1875.
  Fact 0.849: Humboldt was incorporated as a city with a population of 5000 or more on November 7, 2000.
  Fact 0.850: Boa Vista (Paraiba, Brazil) was founded in 1876.
  Fact 0.851: Cajapio (Maranhao, Brazil) was founded in 1876.
  Fact 0.852: Rapid City (South Dakota, United States) was founded in 1876.
  Fact 0.853: Billings (Montana, United States) was founded in 1877.
  Fact 0.854: Jaquimeyes (Barahona, Dominican Republic) was founded in 1877.
  Fact 0.855: Coeur d'Alene (Idaho, United States) was founded in 1878.
  Fact 0.856: Colombo (Parana, Brazil) was founded in 1878.
  Fact 0.857: Vicencia (Pernambuco, Brazil) was founded in 1879.
  Fact 0.858: Vicencia was declared a vila in 1891 and a cidade in 1928.
  Fact 0.859: Pierre (South Dakota, United States) was founded in 1880.
  Fact 0.860: Juneau (Alaska, United States) was founded in 1881.
  Fact 0.861: Moose Jaw (Saskatchewan, Canada) was founded in 1881.
  Fact 0.862: Moose Jaw was incorporated as a city with a population of 5000 or more in November 1903.
  Fact 0.863: Brandon (Manitoba, Canada) was founded in 1882.
  Fact 0.864: Mao (Valverde, Dominican Republic) was founded in 1882.
  Fact 0.865: Restauracion (Dajabon, Dominican Republic) was founded in 1882.
  Fact 0.866: Yorkton (Saskatchewan, Canada) was founded in 1882.
  Fact 0.867: Yorkton was settled by the York Farmers' Colonization Company Limited on May 12, 1882.
  Fact 0.868: Yorkton was incorporated as a city with a population of 5000 or more on February 1, 1928.
  Fact 0.869: Swift Current (Saskatchewan, Canada) was founded in 1882.
  Fact 0.870: The Canadian Pacific Railway arrived at Swift Current on December 10, 1882.
  Fact 0.871: Swift Current was incorporated as a city with a population of 5000 or more on January 15, 1914.
  Fact 0.872: Regina (Saskatchewan, Canada) was founded in 1882.
  Fact 0.873: Regina was first named Pile O' Bones and then renamed Regina.
  Fact 0.874: Regina is the capital city of Saskatchewan.
  Fact 0.875: Regina was incorporated as a city with a population of 5000 or more on June 19, 1903.
  Fact 0.876: Rio Branco (Acre, Brazil) was founded in 1882.
  Fact 0.877: Saskatoon (Saskatchewan, Canada) was founded in 1883.
  Fact 0.878: Saskatoon began as the Temperance Colony founded at Nutana.
  Fact 0.879: Riversdale, Nutana, and Saskatoon were incorporated as a city with a population of 5000 or more in 1906.
  Fact 0.880: Melfort (Saskatchewan, Canada) was founded in 1884.
  Fact 0.881: Melfort's Stoney Creek settlement began in 1884.
  Fact 0.882: Melfort was incorporated as a city with a population of 5000 or more on July 1, 1907.
  Fact 0.883: Vancouver (British Columbia, Canada) was founded in 1886.
  Fact 0.884: Gulfport (Mississippi, United States) was founded in 1887.
  Fact 0.885: Virginia Beach (Virginia, United States) was founded in 1888.
  Fact 0.886: Virginia Beach grew from the Seatack community.
  Fact 0.887: Oklahoma City (Oklahoma, United States) was founded in 1889.
  Fact 0.888: Whiting (Indiana, United States) was founded in 1889.
  Fact 0.889: Ijui (Rio Grande do Sul, Brazil) was founded in 1890.
  Fact 0.890: Aliança (Pernambuco, Brazil) was founded in 1892.
  Fact 0.891: Alianca was declared a vila in 1909 and a cidade in 1928.
  Fact 0.892: Catende (Pernambuco, Brazil) was founded in 1892 as a district of Palmares.
  Fact 0.893: Catende became an autonomous municipality in 1909.
  Fact 0.894: Estevan (Saskatchewan, Canada) was founded in 1892.
  Fact 0.895: The CPR arrived in Estevan in 1893.
  Fact 0.896: Estevan was incorporated as a city with a population of 5000 or more on March 1, 1957.
  Fact 0.897: Guayaramerin (Beni, Bolivia) was founded in 1892.
  Fact 0.898: Restauracion (Dajabon, Dominican Republic) was founded in 1892.
  Fact 0.899: Vernon (British Columbia, Canada) was founded in 1892.
  Fact 0.900: Araripina (Pernambuco, Brazil) was founded in 1893 as a district of Ouricuri.
  Fact 0.901: Araripina became an autonomous municipality in 1928.
  Fact 0.902: Catuipe (Rio Grande do Sul, Brazil) was founded in 1893.
  Fact 0.903: Riberalta (Beni, Bolivia) was founded in 1894.
  Fact 0.904: Traverse City (Michigan, United States) was founded in 1895.
  Fact 0.905: Condado (Pernambuco, Brazil) was founded in 1896.
  Fact 0.906: Condado was declared a cidade in 1958.
  Fact 0.907: Inhumas (Goias, Brazil) was founded in 1896.
  Fact 0.908: Miami (Florida, United States) was founded in 1896.
  Fact 0.909: Santa Catalina la Tinta (Alta Verapaz, Guatemala) was founded in 1896.
  Fact 0.910: Jequie (Bahia, Brazil) was founded in 1897.
  Fact 0.911: Nelson (British Columbia, Canada) was founded in 1897.
  Fact 0.912: Weyburn (Saskatchewan, Canada) was founded in 1898.
  Fact 0.913: Weyburn began as the Signal Hill settlement.
  Fact 0.914: Weyburn was incorporated as a city in 1913.
  Fact 0.915: Chetumal (Quintana Roo, Mexico) was founded in 1898.
  Fact 0.916: Chetumal was originally called Payo Obispo.
  Fact 0.917: Porto Acre (Acre, Brazil) was founded in 1899.
  Fact 0.918: Afranio (Pernambuco, Brazil) was founded in 1900 as a district of Petrolina.
  Fact 0.919: Afranio became an autonomous municipality in 1963.
  Fact 0.920: Fairbanks (Alaska, United States) was founded in 1901.
  Fact 0.921: Florencia (Caqueta, Colombia) was founded in 1902.
  Fact 0.922: Lloydminster (Saskatchewan, Canada) was founded in 1903.
  Fact 0.923: Lloydminster was a Brittania colony settled by Rev. Exton Lloyd and Rev. Isaac Barr.
  Fact 0.924: Lloydminster was incorporated as a city with a population of 5000 or more in 1958.
  Fact 0.925: Cruzeiro do Sul (Acre, Brazil) was founded in 1904.
  Fact 0.926: Grytviken (South Georgia) was founded in 1904.
  Fact 0.927: Sena Madureira (Acre, Brazil) was founded in 1904.
  Fact 0.928: Xapuri (Acre, Brazil) was founded in 1904.
  Fact 0.929: Cobija (Pando, Bolivia) was founded in 1906.
  Fact 0.930: Feijo (Acre, Brazil) was founded in 1906.
  Fact 0.931: Melville (Saskatchewan, Canada) was founded in 1907.
  Fact 0.932: Melville was incorporated as a city with a population of 5000 or more on August 1, 1960.
  Fact 0.933: Tarauaca (Acre, Brazil) was founded in 1907.
  Fact 0.934: Sao Mateus do Sul (Parana, Brazil) was founded in 1908.
  Fact 0.935: Estreito (Maranhao, Brazil) was founded in 1909.
  Fact 0.936: Brasileia (Acre, Brazil) was founded in 1910.
  Fact 0.937: Kindersley (Saskatchewan, Canada) was founded in 1910.
  Fact 0.938: Porto Walter (Acre, Brazil) was founded in 1910.
  Fact 0.939: Villazon (Potosi, Bolivia) was founded in 1910.
  Fact 0.940: Las Vegas (Nevada, United States) was founded in 1911.
  Fact 0.941: Anchorage (Alaska, United States) was founded in 1914.
  Fact 0.942: Yopal (Casanare, Colombia) was founded in 1915.
  Fact 0.943: Leticia (Amazonas, Colombia) was founded in 1928.
  Fact 0.944: Pabellon de Arteaga (Aguascalientes, Mexico) was founded in 1929.
  Fact 0.945: Puerto Carreno (Vichada, Colombia) was founded in 1930.
  Fact 0.946: Loreto (Zacatecas, Mexico) was founded in 1931.
  Fact 0.947: Yellowknife (Northwest Territories, Canada) was founded in 1935.
  Fact 0.948: Mitu (Vaupes, Colombia) was founded in 1935.
  Fact 0.949: San Jose del Guaviare (Guaviare, Colombia) was founded in 1938.
  Fact 0.950: Iqaluit (Nunavut, Canada) was founded in 1942.
  Fact 0.951: Iqaluit was originally called Frobisher Bay.
  Fact 0.952: Freeport (Grand Bahama, Bahamas) was founded in 1955.
  Fact 0.953: Ciudad del Este (Alto Parana, Paraguay) was founded in 1957.
  Fact 0.954: Ciudad del Este was originally called Puerto Flor de Lis.
  Fact 0.955: Rankin Inlet (Nunavut, Canada) was founded in 1957.
  Fact 0.956: Brasilia (Distrito Federal, Brazil) was founded in 1960.
  Fact 0.957: Brasilia was created in 1960 as the national capital.
  Fact 0.958: Belmopan (Cayo, Belize) was founded in 1970.
  Fact 0.959: Linden (Upper Demerara-Berbice, Guyana) was founded in 1970.
  Fact 0.960: Linden was formed by combining the towns of Christianburg, MacKenzie, and Wismar.
  Fact 0.961: Cancun (Quintana Roo, Mexico) was founded in 1970.
  Fact 0.962: Inirida is in Guainia, Colombia.
  Fact 0.963: Palmas (Tocantins, Brazil) was founded 1 year after the creation of the State of Tocantins.
  Fact 0.964: Palmas is the newest and smallest Brazilian capital city.

Retrieved items:
<item 1>
List of cities in the Americas by year of foundation
States 1693 Curitiba Paraná Brazil 1693 Kingston St Andrew Jamaica 1695 Nassau New Providence Bahamas 1695 Saint - Marc Artibonite Haiti 1696 Assú Rio Grande do
Norte Brazil Declared vila in 1766 ; cidade in 1845 . 1696 Rimouski Quebec Canada 1698 Ambato Tungurahua Ecuador 1698 Pensacola Florida United States Known for
being the first European settlement in North America . 1699 Biloxi Mississippi United States 1699 Baton Rouge Louisiana United States 1700 Barreiras Bahia
Brazil 1701 Detroit Michigan United States 1701 São João del Rei Minas Gerais Brazil 1702 Jesús María Aguascalientes Mexico 1702 Mobile Alabama United States
1703 Kaskaskia Illinois United States 1705 Bath North Carolina United States 1706 Albuquerque New Mexico United States 1706 Santo Ângelo Rio Grande do Sul
Brazil 1709 Chihuahua Chihuahua Mexico 1710 Chatham New Jersey United States On land purchased in 1680 .
</item 1>

<item 2>
List of cities in the Americas by year of foundation
1570 Jerez Zacatecas Mexico 1570 Sombrerete Zacatecas Mexico 1571 Cochabamba Cochabamba Bolivia 1572 Huancavelica Huancavelica Peru 1573 San German Puerto Rico
United States Second oldest European settlement in Puerto Rico 1573 Córdoba Córdoba Argentina 1573 Santa Fé Santa Fé Argentina 1573 Buga Valle del Cauca
Colombia 1574 Huaraz Ancash Peru 1574 Tarija Tarija Department Bolivia 1575 Aguascalientes Aguascalientes Mexico 1576 León Guanajuato Mexico 1577 Juayúa
Sonsonate El Salvador 1578 Fort - Liberté Nord - Est Haiti Originally called Bayaha by the Spanish . Oldest continuously inhabited European established
settlement in Haiti . 1578 Tegucigalpa Francisco Morazán Honduras 1579 Coamo Puerto Rico United States Third oldest European settlement in Puerto Rico . In the
same area where the Taínos had had their village of Guayama .
</item 2>

<item 3>
List of cities in the Americas by year of foundation
1710 New Bern North Carolina United States Birthplace of Pepsi 1711 Beaufort South Carolina United States 1711 Ouro Preto Minas Gerais Brazil 1714 Natchitoches
Louisiana United States Oldest settlement in the Louisiana Purchase 1714 Antonina Paraná Brazil 1714 Laguna Santa Catarina Brazil 1714 Serro Minas Gerais Brazil
1715 ( prior to ) Kekionga Indiana United States Capital of the Miami tribe . 1716 Natchez Mississippi United States Dates to the founding of Fort Rosalie by
the French.
</item 3>

<item 4>
List of cities in the Americas by year of foundation
1606 Bayaguana Monte Plata Dominican Republic 1606 Ibarra Imbabura Ecuador 1606 Oruro Oruro Bolivia 1607 Jamestown Virginia United States First permanent
English established settlement in the Americas . 1608 Quebec City Quebec Canada 1609 Fortaleza Ceará Brazil 1610 Kecoughtan Virginia United States 1524 León
León Nicaragua Original location ( now known as the ruins of Leon Viejo ) on the south - western shore of Lago Managua founded in 1524 , but destroyed by
earthquakes , and resettled 20 miles west , in 1610 , to its present - day location at Leon . 1610 Santa Fe New Mexico United States Oldest continuously
inhabited state or territorial capital in the continental United States . ( 8 ) 1610 Harbour Grace Newfoundland and Labrador Canada 1611 Marechal Deodoro ,
Alagoas Alagoas Brazil 1612 São Luís Maranhão Brazil Capital of Maranhão . 1612 St. George 's Bermuda Oldest continuously inhabited English established
settlement in the Americas .
</item 4>

<item 5>
List of cities in the Americas by year of foundation
1564 São Mateus Espírito Santo Brazil 1565 Rio de Janeiro Rio de Janeiro Brazil 1565 Saint Augustine Florida United States Oldest continuously inhabited
European established settlement in the continental United States . Preceded by Puerto Rican colonies , Pensacola , Florida , which was destroyed in 1559 , and
Fort Caroline , destroyed in 1565 . 1565 Tucumán Tucumán Argentina 1567 Caracas Capital District Venezuela 1568 Mazapil Zacatecas Mexico 1568 Goiana Pernambuco
Brazil Declared vila in 1685 ; cidade in 1840 . 1568 Valaparaíso Zacatecas Mexico 1569 Santa Ana Santa Ana El Salvador Built over Mayan city of Sihuatehuacán .
</item 5>

<item 6>
List of cities in the Americas by year of foundation
1643 Dolores Hidalgo Guanajuato Mexico 1644 Salvatierra Guanajuato Mexico 1646 New London Connecticut United States 1648 Alcântara Maranhão Brazil 1648
Paranaguá Paraná Brazil 1649 Annapolis Maryland United States 1650 Castries Castries Saint Lucia 1650 Saint George 's Saint George Grenada 1651 Norwalk
Connecticut United States 1654 Vieux Fort Vieux Fort Saint Lucia 1659 Ciudad Juárez Chihuahua Mexico 1659 El Paso Texas United States 1660 Charlestown Nevis St
Kitts & Nevis 1660 Placentia Newfoundland and Labrador Canada French Capital until 1713 , originally called Plaisance 1660 Rye New York United States 1664
Cayenne French Guiana France Oldest continuously inhabited European established settlement in French Guiana .
</item 6>

<item 7>
List of cities in the Americas by year of foundation
1664 Quibdó Chocó Colombia 1665 Port - de-Paix Nord - Ouest Haiti 1666 Codrington Barbuda Antigua and Barbuda 1666 Newark New Jersey United States 1667 Paraty
Rio de Janeiro Brazil 1668 Sault Ste . Marie Michigan United States ( 13 ) , oldest city in Michigan . 1670 Boqueirão Paraíba Brazil 1670 Cap - Haïtien Nord
Haiti Originally called `` Cap - Français '' 1670 Charleston South Carolina United States 1672 Charlotte Amalie , St. Thomas U.S. Virgin Islands United States
Oldest permanent European settlement in Saint Thomas island . Oldest continuously inhabited Danish established settlement in the Americas . 1673 Kingston
Ontario Canada Grew from Fort Frontenac . Continuously inhabited since 1784 . 1673 San José de Gracia Aguascalientes Mexico 1673 Worcester Massachusetts United
States 1674 Cachoeira Bahia Brazil 1674 Waterbury Connecticut United States 1677 Crosswicks New Jersey United States Quaker settlement . Continuously inhabited
since founding .
</item 7>

<item 8>
List of cities in the Americas by year of foundation
1000 Acoma Pueblo and Taos Pueblo New Mexico United States 1050 Motul Yucatán Mexico 1050 Pachuca Hidalgo Mexico 1100 Cusco Cusco Province Peru 1100 Oraibi
Arizona United States ( Hopi Reservation ) ( 2 ) 1168 Ecatepec Estado de Mexico Mexico 1259 Chimaulhuacan Mexico State Mexico 1313 Xalapa Veracruz Mexico 1325
Tenochtitlan Mexico City Mexico Largest pre-Columbian city in the Americas , later called Mexico City . 1450 Zuni Pueblo New Mexico United States ( 3 ) 1470
Iximche Chimaltenango Guatemala 1492 La Navidad Nord - Est Haiti First European settlement in the New World during the Age of Discovery . 1493 La Isabela Puerto
Plata Dominican Republic 1498 Santo Domingo Distrito Nacional Dominican Republic Capital of the Dominican Republic . Oldest continuously inhabited European
established settlement in the Americas . Founded in 1498 , by Bartholomew Columbus .
</item 8>

<item 9>
List of cities in the Americas by year of foundation
1636 Springfield Massachusetts United States 1636 Providence Rhode Island United States 1637 Hartford Connecticut United States ( 9 ) 1638 Belize City Belize
Belize Oldest continuously inhabited European established settlement in Belize . Original Mayan city called Holzuz . 1638 Fort - de-France Martinique Originally
called Fort Saint Louis . 1638 New Haven Connecticut United States ( 10 ) 1638 Wilmington Delaware United States Grew from Fort Christina , part of the New
Sweden colony . Oldest continuously inhabited Swedish established settlement in the Americas . 1638 Cambridge Massachusetts United States 1639 Rincón de Romos
Aguascalientes Mexico 1639 ( prior to ) St.
</item 9>

<item 10>
List of cities in the Americas by year of foundation
Founded in 1498 , by Bartholomew Columbus . 1502 Santa Cruz del Seibo El Seibo Dominican Republic 1504 Azua de Compostela Azua Dominican Republic 1504 Santiago
de los Caballeros Santiago Dominican Republic 1505 Cotuí Sánchez Ramírez Dominican Republic 1506 Salvaleón de Higüey La Altagracia Dominican Republic 1508 Bonao
Monseñor Nouel Dominican Republic 1508 Caparra Puerto Rico United States First city founded by Europeans , although not continuously inhabited , in the United
States . Abandoned in 1521 with the removal of the capital to San Juan . 1510 Santa Maria la Antigua del Darien Urabá Colombia First city founded by Europeans
on the continent of South America . 1510 Nombre de Dios Colon Panama Oldest continuously inhabited European established settlement in Panama . 1511 Baracoa
Guantánamo Cuba Oldest continuously inhabited European established settlement in Cuba , former capital of Cuba .
</item 10>
=== END ITEM id=bdebe6b175b36c69 ===

=== ITEM id=0a0c0102ca3367b3 ===
Gold sentences and facts:
Sentence 0: The Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League ( NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their fourth NFL title , after winning in 1948 , 1949 , and 1960 .
  Fact 0.0: The Philadelphia Eagles are a professional American football franchise.
  Fact 0.1: The Philadelphia Eagles are based in Philadelphia, Pennsylvania.
  Fact 0.2: The Philadelphia Eagles compete in the National Football League (NFL).
  Fact 0.3: The Philadelphia Eagles are a member club of the league's National Football Conference (NFC) East division.
  Fact 0.4: The Philadelphia Eagles are Super Bowl champions.
  Fact 0.5: The Philadelphia Eagles won Super Bowl LII.
  Fact 0.6: Super Bowl LII was the Eagles' fourth NFL title.
  Fact 0.7: The Eagles had won NFL titles in 1948, 1949, and 1960 before Super Bowl LII.
Sentence 1: Philadelphia Eagles Current season Established July 8 , 1933 ; 84 years ago ( July 8 , 1933 ) First season : 1933 Play in Lincoln Financial Field Philadelphia , Pennsylvania Headquartered in the NovaCare Complex Philadelphia , Pennsylvania Logo Wordmark League / conference affiliations National Football League ( 1933 -- present ) Eastern Division ( 1933 -- 1949 ) American Conference ( 1950 -- 1952 ) Eastern Conference ( 1953 -- 1969 ) Capitol Division ( 1967 -- 1969 ) National Football Conference ( 1970 -- present ) NFC East ( 1970 -- present ) Current uniform Team colors Midnight green , Silver , Black , White Fight song Fly , Eagles Fly Mascot Swoop Personnel Owner ( s ) Jeffrey Lurie Christina Weiss Lurie Chairman Jeffrey Lurie CEO Jeffrey Lurie President Don Smolenski General manager Howie Roseman Head coach Doug Pederson Team history Philadelphia Eagles ( 1933 -- 1942 ; 1944 -- present ) Phil - Pitt `` Steagles '' ( 1943 ) Team nicknames The Birds Championships League championships ( 4 ) NFL championships ( pre-1970 AFL -- NFL merger ) ( 3 ) 1948 , 1949 , 1960 Super Bowl championships ( 1 ) 2017 ( LII ) Conference championships ( 4 ) NFL Eastern : 1960 NFC : 1980 , 2004 , 2017 Division championships ( 13 ) NFL East : 1947 , 1948 , 1949 NFC East : 1980 , 1988 , 2001 , 2002 , 2003 , 2004 , 2006 , 2010 , 2013 , 2017 Playoff appearances ( 25 ) NFL : 1947 , 1948 , 1949 , 1960 , 1978 , 1979 , 1980 , 1981 , 1988 , 1989 , 1990 , 1992 , 1995 , 1996 , 2000 , 2001 , 2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 ) Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present )
  Fact 1.0: The Philadelphia Eagles were established on July 8, 1933.
  Fact 1.1: The Eagles' first season was 1933.
  Fact 1.2: The Eagles play at Lincoln Financial Field in Philadelphia, Pennsylvania.
  Fact 1.3: The Eagles are headquartered in the NovaCare Complex in Philadelphia, Pennsylvania.
  Fact 1.4: The Eagles' team colors are midnight green, silver, black, and white.
  Fact 1.5: The Eagles' fight song is 'Fly, Eagles Fly'.
  Fact 1.6: The Eagles' mascot is Swoop.
  Fact 1.7: The Eagles are owned by Jeffrey Lurie and Christina Weiss Lurie.
  Fact 1.8: Jeffrey Lurie is the chairman of the Eagles.
  Fact 1.9: Jeffrey Lurie is the CEO of the Eagles.
  Fact 1.10: Don Smolenski is the president of the Eagles.
  Fact 1.11: Howie Roseman is the general manager of the Eagles.
  Fact 1.12: Doug Pederson is the head coach of the Eagles.
  Fact 1.13: The Eagles are nicknamed The Birds.
  Fact 1.14: The Eagles have been in the National Football League since 1933.
  Fact 1.15: The Eagles were in the Eastern Division from 1933 to 1949.
  Fact 1.16: The Eagles were in the American Conference from 1950 to 1952.
  Fact 1.17: The Eagles were in the Eastern Conference from 1953 to 1969.
  Fact 1.18: The Eagles were in the Capitol Division from 1967 to 1969.
  Fact 1.19: The Eagles have been in the National Football Conference since 1970.
  Fact 1.20: The Eagles have been in the NFC East since 1970.
  Fact 1.21: The Philadelphia Eagles team existed from 1933 to 1942 and from 1944 to the present.
  Fact 1.22: The Phil-Pitt 'Steagles' team existed in 1943.
  Fact 1.23: The Eagles have won 4 league championships.
  Fact 1.24: The Eagles have won 3 NFL championships before the AFL-NFL merger.
  Fact 1.25: The Eagles won NFL championships in 1948, 1949, and 1960.
  Fact 1.26: The Eagles have won 1 Super Bowl championship.
  Fact 1.27: The Eagles won Super Bowl LII in 2017.
  Fact 1.28: The Eagles have won 4 conference championships.
  Fact 1.29: The Eagles won the NFL Eastern conference championship in 1960.
  Fact 1.30: The Eagles won NFC conference championships in 1980, 2004, and 2017.
  Fact 1.31: The Eagles have won 13 division championships.
  Fact 1.32: The Eagles won NFL East division championships in 1947, 1948, and 1949.
  Fact 1.33: The Eagles won NFC East division championships in 1980, 1988, 2001, 2002, 2003, 2004, 2006, 2010, 2013, and 2017.
  Fact 1.34: The Eagles have made 25 playoff appearances.
  Fact 1.35: The Eagles made NFL playoff appearances in 1947, 1948, 1949, 1960, 1978, 1979, 1980, 1981, 1988, 1989, 1990, 1992, 1995, 1996, 2000, 2001, 2002, 2003, 2004, 2006, 2008, 2009, 2010, 2013, and 2017.
  Fact 1.36: The Eagles played home games at Baker Bowl from 1933 to 1935.
  Fact 1.37: The Eagles played home games at Philadelphia Municipal Stadium in 1936 to 1939 and in 1941.
  Fact 1.38: The Eagles played home games at Connie Mack Stadium in 1940 and from 1942 to 1957.
  Fact 1.39: The Eagles played home games at Franklin Field from 1958 to 1970.
  Fact 1.40: The Eagles played home games at Veterans Stadium from 1971 to 2002.
  Fact 1.41: The Eagles have played home games at Lincoln Financial Field since 2003.

Retrieved items:
<item 1>
Philadelphia Eagles
This would be Philadelphia 's first Super Bowl win and their fourth league championship ( their first championship since 1960 ) . Foles won Super Bowl MVP going
28 -- 43 with 373 passing yards , 3 touchdowns , and an interception . Nick Foles became the first backup quarterback to start and win a Super Bowl since his
opponent Tom Brady won as the backup for Drew Bledsoe in 2001 .
</item 1>

<item 2>
Philadelphia Eagles
External links Wikimedia Commons has media related to Philadelphia Eagles . Official website Philadelphia Eagles Founded in 1933 Based and headquartered in
Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack
Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop Curse of Billy Penn The Lombardi Curse Invincible Philadelphia
Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders Silver Linings Playbook The Garbage Picking Field Goal Kicking Philadelphia Phenomenon Lore
Frankford Yellow Jackets Pennsylvania Keystoners `` Pennsylvania Polka '' Steagles `` Happy Hundred '' Miracle at the Meadowlands Buddy Ball Fog Bowl Bounty
Bowl series Body Bag Game 4th and 26 Miracle at the New Meadowlands Philly Special Rivalries Dallas Cowboys New York Giants Washington Redskins Pittsburgh
Steelers Division championships ( 13 ) 1947 1948 1949 1980 1988 2001 2002 2003 2006 2013 2017 Conference championships ( 4 ) 1960 1980 2017 League championships
( 4 ) 1948 1949 1960 2017 ( LII ) Retired numbers 5
</item 2>

<item 3>
Graph query row:
organization: Philadelphia Eagles
event: Super Bowl LII
date: 2018
score: None
result: None
outcome: None
</item 3>

<item 4>
Philadelphia Eagles
In 2003 , they won the NFC first seed , but Westbrook went down in Week 17 , culminating in a loss to the Carolina Panthers in their 3rd straight NFC
Championship . 2004 season and Super Bowl XXXIX In 2004 , the Philadelphia Eagles had their best season since 1960 , going 13 -- 1 before resting their starters
and losing their next 2 , clinching the 1st seed for the second year in a row . McNabb set career highs , completing 64 % of his passes for 3,875 yards , though
he did n't play all 16 games . McNabb became the first quarterback ever to throw more than 30 touchdowns and fewer than 10 interceptions in a season . His
success could be attributed to the fact that he had a reliable receiver , Terrell Owens , who got 1,200 yards and 14 touchdowns in 14 games . After beating the
Vikings and Falcons the Eagles advanced to Super Bowl XXXIX , where they dueled the New England Patriots .
</item 4>

<item 5>
Super Bowl XXXIX
The Eagles were trying to win their first NFL title since 1960 and the first championship for the city of Philadelphia since Moses Malone 's `` fo ' , fi ' , fo
' '' during the 76ers run to the 1983 NBA Championship . With President George W. Bush being inaugurated for a second time in January , the Eagles were also
trying to end a losing streak -- teams in the city of Philadelphia had lost six straight championships during presidential inauguration years , beginning with
the 76ers loss in 1977 . The streak included the Eagles in Super Bowl XV in 1981 . 1989 was not included in that streak , but 1977 , 1981 , 1985 ( Flyers ) ,
1993 ( Phillies ) , 1997 ( Flyers ) , and 2001 ( 76ers ) were . Television and entertainment ( edit ) The game was televised in the United States by Fox , with
play - by - play announcer Joe Buck and color commentators Cris Collinsworth and Troy Aikman . Pam Oliver ( Patriots sideline ) and Chris Myers ( Eagles
sideline ) served as sideline reporters .
</item 5>

<item 6>
Super Bowl XXXIX
For the Eagles , with the loss , the city of Philadelphia would not see its first championship since the 76ers swept the 1983 NBA Finals until the next time
their city 's teams played for a championship , when the Phillies won the 2008 World Series . The Eagles did not return to the Super Bowl until 2017 , also with
a 13 - 3 record , and defeating the Falcons and Vikings in the playoffs . The Eagles and the Patriots met again in Super Bowl LII , following the 2017 season ,
with the Eagles taking their revenge 41 -- 33 . This is the last time that a team has won back - to - back Super Bowls , with the Patriots losing their second
playoff game a year later and then , for the next eight years , every team either losing their first playoff game or missing them altogether . The streak was
broken in 2014 when the Seahawks defeated the Panthers and ultimately advanced to the Super Bowl , only to lose to a Patriots goal line stand .
</item 6>

<item 7>
Philadelphia Eagles
As of the conclusion of the 2016 season , during the Lurie era , the Eagles are 1 -- 4 in conference championship games and 0 -- 1 in Super Bowls . On August 13
, 2009 , the Eagles signed quarterback Michael Vick . On December 6 , 2009 , Andy Reid became only the 5th coach in NFL history to win 100 or more games with a
single team in a single decade . The other four are Tom Landry , Don Shula , Tony Dungy , and Bill Belichick , all Super Bowl winners . McNabb finally had a
complete receiving corps , between first round draft pick Jeremy Maclin , DeSean Jackson 's 1,000 yard season , and Brent Celek ranking among the top 5 tight
ends in the league . Without Brian Dawkins , defensive end Trent Cole stepped up and became the dominant force on defense with 12 sacks , earning him his second
trip to the Pro Bowl and All - Pro honors .
</item 7>

<item 8>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 8>

<item 9>
Philadelphia Eagles
2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 )
Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present ) The
Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League (
NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their
fourth NFL title , after winning in 1948 , 1949 , and 1960 . The franchise was established in 1933 as a replacement for the bankrupt Frankford Yellow Jackets ,
when a group led by Bert Bell secured the rights to an NFL franchise in Philadelphia .
</item 9>

<item 10>
Philadelphia Eagles
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 10>
=== END ITEM id=0a0c0102ca3367b3 ===
