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

=== ITEM id=cfddf0e1c51eb601 ===
Gold sentences and facts:
Sentence 0: In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers . When Maximoff throws the explosion into the sky with telekinesis , it damages a nearby building , killing several Wakandan humanitarian workers .
  Fact 0.0: In 1991, the brainwashed super-soldier James 'Bucky' Barnes was dispatched from a Hydra base in Siberia.
  Fact 0.1: In 1991, James 'Bucky' Barnes was dispatched to intercept an automobile carrying a case of super-soldier serum.
  Fact 0.2: In the present day, Ultron's defeat in the nation of Sokovia at the hands of the Avengers happened approximately one year earlier.
  Fact 0.3: Ultron was defeated in the nation of Sokovia by the Avengers.
  Fact 0.4: In the present day, Steve Rogers, Natasha Romanoff, Sam Wilson, and Wanda Maximoff stop Brock Rumlow from stealing a biological weapon from a lab in Lagos.
  Fact 0.5: Brock Rumlow blows himself up.
  Fact 0.6: Brock Rumlow hoped to kill Steve Rogers by blowing himself up.
  Fact 0.7: Wanda Maximoff throws the explosion into the sky with telekinesis.
  Fact 0.8: The explosion damages a nearby building.
  Fact 0.9: The explosion killed several Wakandan humanitarian workers.

Retrieved items:
<item 1>
Captain America: Civil War
The next month , it was revealed that the film would be converted to 3D in post-production , and that Brühl would be playing Helmut Zemo . Additionally ,
Elizabeth Olsen revealed she would reprise her role in the film as Wanda Maximoff / Scarlet Witch . Filming ( edit ) Film set for Captain America : Civil War in
Downtown Atlanta Principal photography began on April 27 , 2015 , in Fayette County , Georgia at Pinewood Atlanta Studios , under the working title Sputnik .
Other filming locations in the Atlanta metropolitan area included the Buckhead district of Atlanta ; the Peachtree Christian Church in Midtown Atlanta ;
Downtown Atlanta , including an area near Philips Arena known as The Gulch , which served as the market in Lagos ; Norcross , Georgia ; Porsche 's headquarters
in Atlanta at Aerotropolis Atlanta , which served as the Avengers ' headquarters in the film ; and the Atlanta Civic Center , which served as the IFID (
Institute for Infectious Diseases ) laboratory in Lagos ; the latter 's interiors doubled as MIT , and a third part was used for a Berlin location .
</item 1>

<item 2>
Captain America: Civil War
We wanted to figure out very personal reasons why everyone 's relationship to this idea of registration is going to become complicated . That 's what the
relationship between Steve and Bucky allowed us to do , to get very personal in terms of why people would lean one way or the other . '' Executive producer Nate
Moore added that `` it felt like it was kind of the right time '' to adapt `` Civil War '' given The Avengers , plus many of the Phase Two films ( Thor : The
Dark World , The Winter Soldier , and Avengers : Age of Ultron ) , all dealt with `` world - ending experiences . We felt like we had to tell the next step in
that story , which is ... what is the world 's reaction ? '' In November 2014 , Daniel Brühl joined the cast in an unspecified role , while Anthony Mackie and
Frank Grillo were confirmed to return as Sam Wilson / Falcon and Brock Rumlow / Crossbones , respectively .
</item 2>

<item 3>
Captain America: Civil War
Archived from the original on October 30 , 2016 . Captain America : Civil War Captain America Joe Simon Jack Kirby Alter egos Steve Rogers Isaiah Bradley
William Nasland Jeffrey Mace William Burnside John Walker James Buchanan Barnes Sam Wilson Supporting characters Rikki Barnes Battlestar Black Widow Bucky Peggy
Carter Sharon Carter Demolition Man ( D - Man ) Diamondback Falcon Jack Flag Free Spirit Golden Girl Howling Commandos Invaders Nuke Redwing Rick Jones Nomad
Jack Monroe Enemies A.I.M. Aleksander Lukin Armadillo Arnim Zola Baron Blood Baron Strucker Baron Zemo Heinrich Zemo Helmut Zemo Batroc the Leaper Blackwing
Circus of Crime Crossbones Doctor Faustus Doughboy Flag - Smasher Grand Director Hate - Monger Hydra Master Man MODOK Protocide Red Skull Scourge of the
Underworld Secret Empire Serpent Society Serpent Squad Sin Slug Superia Viper Madame Hydra Watchdogs Publications Captain America Captain America ( vol.
</item 3>

<item 4>
Captain America: Civil War
Contents ( hide ) 1 Plot 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 3.4. 1 Visual effects 4 Music 5 Release 5.1
Marketing 5.1. 1 Merchandise 5.2 Home media 6 Reception 6.1 Box office 6.1. 1 United States and Canada 6.1. 2 Other territories 6.2 Critical response 6.3
Accolades 7 Future 8 Notes 9 References 10 External links Plot ( edit ) In 1991 , the brainwashed super-soldier James `` Bucky '' Barnes is dispatched from a
Hydra base in Siberia to intercept an automobile carrying a case of super-soldier serum . In the present day , approximately one year after Ultron 's defeat in
the nation of Sokovia at the hands of the Avengers , Steve Rogers , Natasha Romanoff , Sam Wilson , and Wanda Maximoff stop Brock Rumlow from stealing a
biological weapon from a lab in Lagos . Rumlow blows himself up , hoping to kill Rogers .
</item 4>

<item 5>
Captain America: Civil War
In one program , Everhart discusses events leading to Captain America : Civil War . The post-credits scene for Ant - Man featured footage shot by the Russo
brothers for Captain America : Civil War , showing Wilson and Rogers with Barnes in their custody , and unable to contact Stark because of `` the accords '' ;
Wilson mentions that he `` know ( s ) a guy '' , implying Lang . Feige explained that when the post-credit sequence appears in Civil War it may appear with ``
different takes ... different angles . '' Stan said the scene was shot in early May 2015 , and would appear in the middle of Civil War . The first footage of
the film debuted in August 2015 at the D23 Expo . Additional footage was shown in September 2015 at the Asia Pop Comic Convention . The footage shown at the D23
Expo and 2015 Asia Pop Comic Convention received an overwhelmingly positive response from the audiences . The first trailer for the film debuted on Jimmy Kimmel
Live ! on November 25 , 2015 , and within hours became the number one trending topic on Twitter .
</item 5>

<item 6>
Captain America: Civil War
Satisfied that he got Stark and Rogers to fight each other , Zemo attempts suicide , but T'Challa stops him and takes him to the authorities . In the aftermath
, Stark provides Rhodes with exoskeletal leg braces that allow him to walk again , while Rogers breaks his allies out of the Raft . In a mid-credits scene ,
Barnes , granted asylum in Wakanda , chooses to return to cryogenic sleep until a cure for his brainwashing is found . In a post-credits scene , Parker tests a
new gadget built by Stark . Cast ( edit ) ( L-R ) Nate Moore , Kevin Feige , the Russo brothers , Paul Bettany , Daniel Brühl , Robert Downey , Jr. , Chris
Evans , Tom Holland , Paul Rudd , Elizabeth Olsen , Jeremy Renner , Anthony Mackie , Sebastian Stan , and Emily VanCamp at the London premiere Chris Evans as
Steve Rogers / Captain America : The leader of a faction of Avengers against regulation ; a World War II veteran who was enhanced to the peak of human
physicality and frozen in suspended animation , before waking up in the modern world .
</item 6>

<item 7>
Captain America: Civil War
Captain America : Civil War - Wikipedia Captain America : Civil War Jump to : navigation , search Captain America : Civil War Theatrical release poster Directed
by Anthony Russo Joe Russo Produced by Kevin Feige Screenplay by Christopher Markus Stephen McFeely Based on Captain America by Joe Simon Jack Kirby Starring
Chris Evans Robert Downey Jr . Scarlett Johansson Sebastian Stan Anthony Mackie Don Cheadle Jeremy Renner Chadwick Boseman Paul Bettany Elizabeth Olsen Paul
Rudd Emily VanCamp Tom Holland Frank Grillo William Hurt Daniel Brühl Music by Henry Jackman Cinematography Trent Opaloch Edited by Jeffrey Ford Matthew Schmidt
Production company Marvel Studios Distributed by Walt Disney Studios Motion Pictures Release date April 12 , 2016 ( 2016 - 04 - 12 ) ( Dolby Theatre ) May 6 ,
2016 ( 2016 - 05 - 06 ) ( United States ) Running time 147 minutes Country United States Language English Budget $250 million Box office $1.153 billion Captain
America : Civil War is a 2016 American superhero film based on the Marvel Comics character Captain America , produced by Marvel Studios and distributed by Walt
Disney Studios Motion Pictures .
</item 7>

<item 8>
Captain America: Civil War
Following the November 2014 hacking of Sony Pictures ' computers , emails between Sony Pictures Entertainment co-chairman Amy Pascal and president Doug Belgrad
were released stating that Marvel wanted to include Spider - Man ( whose film rights are licensed to Sony ) in the film , but talks between the studios
concerning this were believed to have broken down . However , in February 2015 , Sony Pictures and Marvel Studios reached a licensing deal for the use of Spider
- Man in an MCU film , and reports indicated that the character would indeed appear in Civil War . The Russos stated they were lobbying for months to include
the character in the film . In January 2015 , Mackie revealed that , in addition to Atlanta , filming locations would include Puerto Rico and Berlin , while the
Russo brothers confirmed that Scarlett Johansson would return in the film as Natasha Romanoff / Black Widow . Editor Jeffrey Ford , who worked on The Winter
Soldier , also signed on for Civil War . In March 2015 , Jeremy Renner was revealed to be reprising his role as Clint Barton / Hawkeye .
</item 8>

<item 9>
Captain America: Civil War
It is the sequel to 2011 's Captain America : The First Avenger and 2014 's Captain America : The Winter Soldier , and the thirteenth film in the Marvel
Cinematic Universe ( MCU ) . The film is directed by Anthony and Joe Russo , with a screenplay by Christopher Markus & Stephen McFeely , and features an
ensemble cast , including Chris Evans , Robert Downey Jr. , Scarlett Johansson , Sebastian Stan , Anthony Mackie , Don Cheadle , Jeremy Renner , Chadwick
Boseman , Paul Bettany , Elizabeth Olsen , Paul Rudd , Emily VanCamp , Tom Holland , Frank Grillo , William Hurt , and Daniel Brühl . In Captain America : Civil
War , disagreement over international oversight of the Avengers fractures them into opposing factions -- one led by Steve Rogers and the other by Tony Stark .
Development of Civil War began in late 2013 when Markus and McFeely began writing the screenplay , which borrows concepts from the 2006 comic book storyline ``
Civil War '' , while also focusing on story and character elements from the previous Captain America films to conclude the trilogy .
</item 9>

<item 10>
Captain America: Civil War
Following positive reactions to test screenings of The Winter Soldier , the Russo brothers were brought back to direct in early 2014 . The film 's title and
premise were revealed in October 2014 , along with Downey 's casting ; additional cast members joined in the following months . Principal photography began in
April 2015 at Pinewood Atlanta Studios in Fayette County , Georgia , and continued in the Metro Atlanta area before concluding in Germany in August 2015 , with
the film being the first to use IMAX 's digital 2D cameras ( for the film 's central airport fight sequence ) . Visual effects were provided by nearly 20
different studios during the post-production process . Captain America : Civil War held its world premiere in Los Angeles on April 12 , 2016 , and was released
in the United States on May 6 , 2016 , in 3D and IMAX 3D . The film became a critical and commercial success , grossing over $1.1 billion worldwide , making it
the highest - grossing film of 2016 and , at the time , the 12th - highest - grossing film ever .
</item 10>
=== END ITEM id=cfddf0e1c51eb601 ===

=== ITEM id=30aa016716dd019a ===
Gold sentences and facts:
Sentence 0: The soma ( cell body ) of each pseudounipolar neuron is located within a dorsal root ganglion . The axon leaves the cell body ( and out of the dorsal root ganglion ) into the dorsal root , where it splits into two branches . The central branch goes to the posterior ( dorsal ) horn of the spinal cord , where it forms synapses with other neurons . The peripheral branch travels through the distal dorsal root into the spinal nerve all the way until skin , joint , and muscle .
  Fact 0.0: The soma (cell body) of each pseudounipolar neuron is located within a dorsal root ganglion.
  Fact 0.1: The axon leaves the cell body.
  Fact 0.2: The axon leaves the dorsal root ganglion.
  Fact 0.3: The axon goes into the dorsal root.
  Fact 0.4: In the dorsal root, the axon splits into two branches.
  Fact 0.5: The central branch of the axon goes to the posterior (dorsal) horn of the spinal cord.
  Fact 0.6: The central branch forms synapses with other neurons.
  Fact 0.7: The peripheral branch travels through the distal dorsal root.
  Fact 0.8: The peripheral branch travels into the spinal nerve.
  Fact 0.9: The peripheral branch travels all the way until skin, joint, and muscle.

Retrieved items:
<item 1>
Motor neuron
the command of visceral muscles is disynaptic ( involving two neurons : the general visceral motor neuron located in the CNS , which synapses onto a ganglionic
neuron , located in the PNS , which synapses onto the muscle ) . All vertebrate motor neurons are cholinergic , that is , they release the neurotransmitter
acetylcholine . Parasympathetic ganglionic neurons are also cholinergic , whereas most sympathetic ganglionic neurons are noradrenergic , that is , they release
the neurotransmitter noradrenaline . ( see Table ) Motor neurons in Regenerative medicine ( edit ) Human lower motor neurons can be generated in vitro from
embryonic stem cells and induced pluripotent stem cells . See also ( edit ) Betz cell Central chromatolysis Motor dysfunction Nerve References ( edit ) Jump up
^ Tortora , Gerard ; Derrickson , Bryan ( 2014 ) . Principles of Anatomy & Physiology ( 14th ed . ) . New Jersey : John Wiley & Sons , Inc . pp. 406 , 502 , 541
. ISBN 978 - 1 - 118 - 34500 - 9 .
</item 1>

<item 2>
Pseudounipolar neuron
Pseudounipolar neuron - wikipedia Pseudounipolar neuron Jump to : navigation , search Pseudounipolar neuron 1 : Pseudounipolar neuron , 2 : Bipolar neuron
Details System Nervous system Identifiers Latin neuron pseudounipolare TH H2. 00.06. 1.00047 Anatomical terms of neuroanatomy ( edit on Wikidata ) A
pseudounipolar neuron ( pseudo -- false , uni -- one ) is a kind of sensory neuron in the peripheral nervous system . This neuron contains an axon that has
split into two branches ; one branch runs to the periphery and the other to the spinal cord . Contents ( hide ) 1 Structure 1.1 In the dorsal root ganglia 1.2
In the sensory ganglia of cranial nerves 2 Function 2.1 Signal pathway 3 See also 4 References Structure ( edit ) By definition , a pseudounipolar neuron has
one axon with two branches : central and peripheral . These axonal branches should not be confused with dendrites .
</item 2>

<item 3>
Pseudounipolar neuron
Ganglion Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion ( Preganglionic nerve fibers Postganglionic nerve fibers ) Connective tissues
Epineurium Perineurium Endoneurium Nerve fascicle Neuroglia Myelination : Schwann cell Neurilemma Myelin incisure Node of Ranvier Internodal segment Satellite
glial cell Neurons / nerve fibers Parts Soma Axon hillock Axon Telodendron Axon terminals Axoplasm Axolemma Neurofibril / neurofilament Dendrite Nissl body
Dendritic spine Apical dendrite / Basal dendrite Types Bipolar Unipolar Pseudounipolar Multipolar Interneuron Renshaw Afferent nerve fiber / Sensory neuron GSA
GVA SSA SVA fibers Ia or Aα Ib or Golgi or Aα II or Aβ and Aγ III or Aδ or fast pain IV or C or slow pain Efferent nerve fiber / Motor neuron GSE GVE SVE Upper
motor neuron Lower motor neuron α motorneuron β motorneuron γ motorneuron Termination Synapse Electrical synapse / Gap junction Chemical
</item 3>

<item 4>
Motor neuron
segment Satellite glial cell Neurons / nerve fibers Parts Soma Axon hillock Axon Telodendron Axon terminals Axoplasm Axolemma Neurofibril / neurofilament
Dendrite Nissl body Dendritic spine Apical dendrite / Basal dendrite Types Bipolar Unipolar Pseudounipolar Multipolar Interneuron Renshaw Afferent nerve fiber /
Sensory neuron GSA GVA SSA SVA fibers Ia or Aα Ib or Golgi or Aα II or Aβ and Aγ III or Aδ or fast pain IV or C or slow pain Efferent nerve fiber / Motor neuron
GSE GVE SVE Upper motor neuron Lower motor neuron α motorneuron β motorneuron γ motorneuron Termination Synapse Electrical synapse / Gap junction Chemical
synapse Synaptic vesicle Active zone Postsynaptic density Autapse Ribbon synapse Neuromuscular junction Sensory receptors Meissner 's corpuscle Merkel nerve
ending Pacinian corpuscle Ruffini ending Muscle spindle Free nerve ending Nociceptor Olfactory receptor neuron Photoreceptor cell Hair cell Taste bud Retrieved
from `` https://en.wikipedia.
</item 4>

<item 5>
Pseudounipolar neuron
These axonal branches should not be confused with dendrites . These sensory neurons are an exception to the typical neuron , in that they do not have separate
dendrites and an axonal process , but rather one branched process that serves both functions . The axon has a peripheral branch ( from the cell body to the
periphery : skin , joint and muscle ) and a central branch ( from the cell body to spinal cord ) . In the dorsal root ganglia ( edit ) The soma ( cell body ) of
each pseudounipolar neuron is located within a dorsal root ganglion . The axon leaves the cell body ( and out of the dorsal root ganglion ) into the dorsal root
, where it splits into two branches . The central branch goes to the posterior ( dorsal ) horn of the spinal cord , where it forms synapses with other neurons .
The peripheral branch travels through the distal dorsal root into the spinal nerve all the way until skin , joint , and muscle .
</item 5>

<item 6>
Motor neuron
These columns are further explained in the table below . Motor Columns of Spinal Cord Motor Column Location in Spinal Cord Target Median Motor Column Present
entire length Axial muscles Hypaxial Motor Column Thoracic Region Body wall muscles Preganglionic Motor Column Thoracic Region Sympathetic ganglion Laterl Motor
Column Brachial and Lumbar region ( both regions are further divided into medial and lateral domains ) Muscles of the limbs Phrenic Motor Coulmn Cervical Region
Diaphragm Anatomy and Physiology ( edit ) Spinal cord tracts Location of lower motor neurons in spinal cord Upper motor neurons ( edit ) Upper motor neurons
originate in the motor cortex located in the precentral gyrus . The cells that make up the primary motor cortex are Betz cells , which are a type of pyramidal
cell . The axons of these cells descend from the cortex to form the corticospinal tract . White matter Tract ( edit ) White matter tracts are bundles of axons
and serve as means to propagate nerve impulses on a large scale . These are found in the spinal cord and appear white from the myelin surrounding axons .
</item 6>

<item 7>
Motor neuron
Branch of NS Position Neurotransmitter Somatic n / a Acetylcholine Parasympathetic Preganglionic Acetylcholine Parasympathetic Ganglionic Acetylcholine
Sympathetic Preganglionic Acetylcholine Sympathetic Ganglionic Norepinephrine * * Except fibers to sweat glands and certain blood vessels Motor neuron
neurotransmitters General visceral motor neurons ( edit ) These motor neurons indirectly innervate cardiac muscle and smooth muscles of the viscera ( the
muscles of the arteries ) : they synapse onto neurons located in ganglia of the autonomic nervous system ( sympathetic and parasympathetic ) , located in the
peripheral nervous system ( PNS ) , which themselves directly innervate visceral muscles ( and also some gland cells ) . In consequence : the motor command of
skeletal and branchial muscles is monosynaptic ( involving only one motor neuron , respectively , somatic and branchial , which synapses onto the muscle ) .
</item 7>

<item 8>
Pseudounipolar neuron
. Cambridge : McGrawHill . p. 24 . ISBN 978 - 0838577011 . ^ Jump up to : MacCallum , Don . `` Peripheral Nervous System '' . Histology and Virtual Microscopy
Learning Resources . University of Michigan Medical School . Archived from the original on 7 July 2014 . Retrieved 24 June 2014 . Jump up ^ Miller 's Anatomy of
the Dog , By Howard E. Evans , Alexander de Lahunta , 4th ed , pp 565 ( hide ) Nervous tissue CNS Tissue Types Grey matter White matter Projection fibers
Association fiber Commissural fiber Lemniscus Funiculus Fasciculus Nerve tract Decussation Commissure Neuropil Meninges Cell Types Neuronal Pyramidal Purkinje
Granule Glial insulating : Myelination : Oligodendrocyte other Astrocyte Radial glial cell Ependymal cells Tanycyte Microglia PNS General Dorsal Root Ganglion
Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion (
</item 8>

<item 9>
Pseudounipolar neuron
In the sensory ganglia of cranial nerves ( edit ) Pseudounipolar neurons are also found in the sensory ganglia of cranial nerves V , VII , IX and X , all mixed
- cranial nerves . The vestibulocochlear nerve , has the bipolar neuron type in both the spiral ganglion as well as Scarpa 's ganglion . Function ( edit )
Signal pathway ( edit ) These neurons have sensory receptors on skin , joints , muscles , and other parts of the body . The area of the axon that is closest to
the receptor is the trigger zone for the neuron . The signal is conducted through the axon to the dorsal root ganglion 's cell body , then through the axon and
ending at the sensory nuclei in the posterior horn of the spinal cord . See also ( edit ) Bipolar neuron Multipolar neuron Unipolar neuron References ( edit ) ^
Jump up to : Kandel , Eric R. ; Schwartz , James H. ; Mack , Thomas M. Jessell ( 2000 ) . Principles of neural Science ( 4th ed . ) . Cambridge : McGrawHill .
p. 24 .
</item 9>

<item 10>
Motor neuron
; Mallatt , J. ( 1997 ) . Human Anatomy ( 2nd ed . ) . Menlo Park , CA : Benjamin / Cummings . ISBN 0 - 8053 - 4068 - 8 . ( hide ) Nervous tissue CNS Tissue
Types Grey matter White matter Projection fibers Association fiber Commissural fiber Lemniscus Funiculus Fasciculus Nerve tract Decussation Commissure Neuropil
Meninges Cell Types Neuronal Pyramidal Purkinje Granule Glial insulating : Myelination : Oligodendrocyte other Astrocyte Radial glial cell Ependymal cells
Tanycyte Microglia PNS General Dorsal Root Ganglion Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion ( Preganglionic nerve fibers
Postganglionic nerve fibers ) Connective tissues Epineurium Perineurium Endoneurium Nerve fascicle Neuroglia Myelination : Schwann cell Neurilemma Myelin
incisure Node of Ranvier Internodal segment Satellite glial cell Neurons / nerve fibers Parts Soma Axon hillock Axon
</item 10>
=== END ITEM id=30aa016716dd019a ===

=== ITEM id=e49af80011506a0a ===
Gold sentences and facts:
Sentence 0: From September 1947 , Toyota 's small - sized vehicles were sold under the name `` Toyopet '' ( トヨペット ) . The first vehicle sold under this name was the Toyopet SA , but it also included vehicles such as the Toyopet SB light truck , Toyopet Stout light truck , Toyopet Crown , Toyopet Master , and the Toyopet Corona . The word `` Toyopet ( Japanese article ) '' was a nickname given to the Toyota SA due to its small size , as the result of a naming contest the Toyota Company organized in 1947 . However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The name was soon dropped for the American market , but continued in other markets until the mid-1960s .
  Fact 0.0: From September 1947, Toyota's small-sized vehicles were sold under the name 'Toyopet' (トヨペット).
  Fact 0.1: The first vehicle sold under the name Toyopet was the Toyopet SA.
  Fact 0.2: Toyopet vehicles also included the Toyopet SB light truck.
  Fact 0.3: Toyopet vehicles also included the Toyopet Stout light truck.
  Fact 0.4: Toyopet vehicles also included the Toyopet Crown.
  Fact 0.5: Toyopet vehicles also included the Toyopet Master.
  Fact 0.6: Toyopet vehicles also included the Toyopet Corona.
  Fact 0.7: The word 'Toyopet' was a nickname given to the Toyota SA due to its small size.
  Fact 0.8: The Toyopet nickname resulted from a naming contest the Toyota Company organized in 1947.
  Fact 0.9: When Toyota entered the American market in 1957 with the Crown, the Toyopet name was not well received due to connotations of toys and pets.
  Fact 0.10: The Toyopet name was soon dropped for the American market.
  Fact 0.11: The Toyopet name continued in other markets until the mid-1960s.

Retrieved items:
<item 1>
Toyota
Manufacturing France Toyota Motor Manufacturing Turkey Toyota Motor North America Toyota Motor Engineering & Manufacturing North America Toyota Motor
Philippines Toyota Motor Thailand Toyota Motorsport GmbH Joint ventures WiLL GAC Toyota ( 50 % ) Indus Motors Company Sichuan FAW Toyota Motor ( 50 % ) Toyota
Kirloskar Motor Toyota Peugeot Citroën Automobile Czech UMW Toyota Motor ( 49 % ) Current models Cars 86 Agya Allion Auris Aurion Avalon Avensis Aygo Camry
Camry Hybrid Corolla Corolla Altis Corolla Axio Corolla Fielder Corolla iM Crown Crown Athlete Crown Hybrid Crown Majesta Crown Royal Etios Mark X Mirai Passo
Pixis Epoch Pixis Joy Pixis Mega Premio Porte Prius Prius Plug - in Hybrid Prius c Probox Roomy Sai Succeed Tank Verso Vios Vitz Yaris Yaris iA Pickup trucks /
Utilities Hilux Hilux Vigo Hilux Revo Land Cruiser 79 Series Pixis Truck Tacoma Tundra SUVs / Crossovers 4Runner C - HR Fortuner Harrier Harrier Hybrid
Highlander Kluger Highlander Hybrid Kluger Hybrid Land Cruiser 200 Land Cruiser Prado Land Cruiser 74 Series Land Cruiser 76 Series
</item 1>

<item 2>
Toyota
The sales price was 3,350 yen , 400 yen cheaper than Ford or GM cars . House of Toyota founder Kiichiro Toyoda , near Toyota City In September 1936 , the
company ran a public competition to design a new logo . Of 27,000 entries , the winning entry was the three Japanese katakana letters for `` Toyoda '' in a
circle . But Rizaburo Toyoda , who had married into the family and was not born with that name , preferred `` Toyota '' ( トヨタ ) because it took eight brush
strokes ( a lucky number ) to write in Japanese , was visually simpler ( leaving off the diacritic at the end ) , and with a voiceless consonant instead of a
voiced one ( voiced consonants are considered to have a `` murky '' or `` muddy '' sound compared to voiceless consonants , which are `` clear '' ) . Inside the
house of Toyota founder Kiichiro Toyoda , near Toyota City Since toyoda literally means `` fertile rice paddies '' , changing the name also prevented the
company from being associated with old - fashioned farming .
</item 2>

<item 3>
Toyota
However , when Toyota eventually entered the American market in 1957 with the Crown , the name was not well received due to connotations of toys and pets . The
name was soon dropped for the American market , but continued in other markets until the mid-1960s . 1960s -- 1970s ( edit ) By the early 1960s , the US had
begun placing stiff import tariffs on certain vehicles . The so - called `` chicken tax '' of 1964 placed a 25 % tax on imported light trucks . In response to
the tariff , Toyota , Nissan Motor Co. and Honda Motor Co. began building plants in the US by the early 1980s . 1980s ( edit ) With over 40 million sold , the
Corolla is one of the most popular and best selling cars in the world . Toyota received its first Japanese Quality Control Award at the start of the 1980s and
began participating in a wide variety of motorsports . Due to the 1973 oil crisis , consumers in the lucrative US market began turning to small cars with better
fuel economy .
</item 3>

<item 4>
Toyota
In Chinese - speaking countries using simplified Chinese characters ( e.g. China ) , Toyota is known as `` 丰田 '' ( pronounced as Fēngtián in Mandarin Chinese
and Hɔng Tshan in Minnanese ) . These are the same characters as the founding family 's name `` Toyoda '' in Japanese . Logo until 1989 , now still used as a
corporate logo Logo since 1989 Japan ( edit ) Toyota Store Aichi Toyopet Store , Saitama Corolla Store Kyotanabe Netz Store , Tsurumi - ku , Osaka Wikimedia
Commons has media related to Toyota dealerships . In Japan , Toyota currently maintains separate dealership sales channels . The first sales channel established
in 1946 called `` Toyota Store '' ( トヨタ 店 Toyota Mise ) sells large luxury sedans such as the Toyota Century , and the Toyota Crown . In 1955 , the `` Toyopet
Store '' ( トヨペット 店 Toyopetto - ten ) arrived , originally established to sell the Toyota Corona and the Toyopet ToyoAce truck .
</item 4>

<item 5>
Toyota
The newly formed word was trademarked and the company was registered in August 1937 as the Toyota Motor Company . The Supra JZA80 has now become an automobile
icon the world over . Rivaling Nissan 's GTR . The MR2 SW20 , a mid-engine sports car from the 1990s . First - generation Toyopet Crown Model RSD ( 1955 / 1 --
1958 / 10 ) Toyota at the Rally Dakar , 1992 1940s -- 1950s ( edit ) From September 1947 , Toyota 's small - sized vehicles were sold under the name `` Toyopet
'' ( トヨペット ) . The first vehicle sold under this name was the Toyopet SA , but it also included vehicles such as the Toyopet SB light truck , Toyopet Stout
light truck , Toyopet Crown , Toyopet Master , and the Toyopet Corona . The word `` Toyopet ( Japanese article ) '' was a nickname given to the Toyota SA due to
its small size , as the result of a naming contest the Toyota Company organized in 1947 .
</item 5>

<item 6>
Toyota
Inside Investor . Retrieved July 24 , 2013 . Jump up ^ `` The Story of Sakichi Toyoda '' . Toyota Industries Corporation . Retrieved September 22 , 2017 . Jump
up ^ `` History Of Toyota '' . Toyota . Retrieved August 15 , 2011 . Jump up ^ `` Toyota Company History from 1867 to 1939 '' . Toyota . Archived from the
original on May 10 , 2013 . Retrieved September 11 , 2010 . Jump up ^ `` Information from a sign at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi
Pref '' . Toyota . Archived from the original on November 20 , 2012 . Retrieved December 7 , 2012 . Jump up ^ Schmitt , Bertel ( July 5 , 2011 ) . `` At Home
With The Toyodas -- And A Mysterious Cherry Tree '' . Thetruthaboutcars . Retrieved August 15 , 2011 . Jump up ^ Davis , Pedr ( 1999 ) .
</item 6>

<item 7>
Toyota
Toyota - wikipedia Toyota Jump to : navigation , search For other uses , see Toyota ( disambiguation ) . Toyota Motor Corporation hkl headquarters in Toyota
City , Japan Type Public ( K.K. ) Traded as TYO : 7203 LSE : TYT NYSE : TM TOPIX Core 30 Component Industry Automotive Founded August 28 , 1937 ; 80 years ago (
1937 - 08 - 28 ) Headquarters Toyota , Aichi , Japan Area served Worldwide Key people Kiichiro Toyoda ( Chairman ) Shigeru Hayakawa ( Vice Chairman ) Takeshi
Uchiyamada ( President ) Products Automobiles , luxury vehicles , commercial vehicles , engines Production output 16,985,870 units ( CY 2015 ) Services Banking
, financing , leasing Revenue ¥ 28.40 trillion ( 2016 ) Operating income ¥ 2.85 trillion ( 2016 ) Profit ¥ 2.31 trillion ( 2016 ) Total assets ¥ 47.42 trillion
( 2016 ) Total equity ¥ 17.22 trillion ( 2016 ) Owner Toyota Industries ( 6.73 % ) Nippon Life ( 3.
</item 7>

<item 8>
Toyota
Toyota is publicly traded on the Tokyo , Osaka , Nagoya , Fukuoka , and Sapporo exchanges under company code TYO : 7203 . In addition , Toyota is foreign -
listed on the New York Stock Exchange under NYSE : TM and on the London Stock Exchange under LSE : TYT . Toyota has been publicly traded in Japan since 1949 and
internationally since 1999 . As reported on its consolidated financial statements , Toyota has 540 consolidated subsidiaries and 226 affiliates . Toyota Motor
North America ( 100 % -- 2004 ) Toyota Canada Inc. ( 51 % -- 2013 ) Toyota Tsusho -- Trading company for the Toyota Group Daihatsu Motor Company ( 100 % ) Hino
Motors ( 50.1 % -- 2001 ) Lexus 100 % ( 1989 ) Scion 100 % ( 2003 ) DENSO ( 24.74 % -- September 30 , 2006 ) Toyota Industries ( 23.51 % -- March 31 , 2006 )
Aisin Seiki Co .
</item 8>

<item 9>
Toyota
Toyota Technical Center , Ann Arbor Charter Township , Michigan Toyota has a large presence in the United States with six major assembly plants in Huntsville ,
Alabama , Georgetown , Kentucky , Princeton , Indiana , San Antonio , Texas , Buffalo , West Virginia , and Blue Springs , Mississippi . Toyota had a joint -
venture operation with General Motors at New United Motor Manufacturing Inc. in Fremont , California , which began in 1984 and ended in 2009 . It still has a
joint venture with Subaru at Subaru of Indiana Automotive , Inc. in Lafayette , Indiana , which started in 2006 . In these assembly plants , the Camry and the
Tundra are manufactured , among others . Toyota marketing , sales , and distribution in the US are conducted through a separate subsidiary , Toyota Motor Sales
, U.S.A. , Inc . It has started producing larger trucks , such as the new Tundra , to go after the large truck market in the United States . Toyota is also
pushing hybrid vehicles in the US such as the Prius , Camry Hybrid , Highlander Hybrid , and various Lexus products . Currently , Toyota has no plans to offer
diesel motor options in its North American products , including the light - duty pickup trucks .
</item 9>

<item 10>
Toyota
The principle of jidoka , which means the machine stops itself when a problem occurs , became later a part of the Toyota Production System . Looms were built on
a small production line . In 1929 , the patent for the automatic loom was sold to the British company Platt Brothers , generating the starting capital for the
automobile development . Mass production of Toyoda automated loom , displayed at the Toyota Museum in Nagakute - cho , Aichi - gun , Aichi Pref . Japan The
production of Toyota automobiles was started in 1933 as a division of Toyoda Automatic Loom Works devoted to the production of automobiles under the direction
of the founder 's son , Kiichiro Toyoda . Its first vehicles were the A1 passenger car and the G1 in 1935 . The Toyota Motor Co. was established as an
independent company in 1937 Toyoda Standard Sedan AA 1936 Vehicles were originally sold under the name `` Toyoda '' ( トヨダ ) , from the family name of the
company 's founder , Kiichirō Toyoda . In April 1936 , Toyoda 's first passenger car , the Model AA , was completed .
</item 10>
=== END ITEM id=e49af80011506a0a ===

=== ITEM id=c8bacd8f0ff3f207 ===
Gold sentences and facts:
Sentence 0: The coastline of the island consists of a combination of long beaches , dunes , red sandstone cliffs , salt water marshes and numerous bays and harbours . The beaches , dunes and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration which oxidizes upon exposure to the air . The geological properties of a white silica sand found at Basin Head are unique in the province ; the sand grains cause a scrubbing noise as they rub against each other when walked on . Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours . The magnificent sand dunes at Greenwich are of particular significance . The shifting , parabolic dune system is home to a variety of birds and rare plants and is also a site of significant archeological interest .
  Fact 0.0: The coastline of the island consists of a combination of long beaches, dunes, red sandstone cliffs, salt water marshes, and numerous bays and harbours.
  Fact 0.1: The beaches, dunes, and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration.
  Fact 0.2: That iron concentration oxidizes upon exposure to air.
  Fact 0.3: The geological properties of a white silica sand found at Basin Head are unique in the province.
  Fact 0.4: The sand grains at Basin Head cause a scrubbing noise as they rub against each other when walked on.
  Fact 0.5: Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours.
  Fact 0.6: The magnificent sand dunes at Greenwich are of particular significance.
  Fact 0.7: The shifting, parabolic dune system at Greenwich is home to a variety of birds and rare plants.
  Fact 0.8: The shifting, parabolic dune system at Greenwich is also a site of significant archeological interest.

Retrieved items:
<item 1>
Sub-Saharan Africa
European Plain Eurasian Steppe Pontic -- Caspian steppe Wild Fields Pannonian Basin Great Hungarian Plain Little Hungarian Plain Eastern Slovak Lowland Regions
of North America Northern Eastern Canada Western Canada Canadian Prairies Central Canada Northern Canada Atlantic Canada The Maritimes French Canada English
Canada Acadia Acadian Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser 's Triangle Canadian Shield Interior Alaska - Yukon
lowland taiga Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic Archipelago Labrador Peninsula Gaspé Peninsula Avalon
Peninsula Bay de Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula ( Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay
Hudson Bay James Bay Greenland Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic Commonwealth West Midwest Upper Midwest Mountain States
Intermountain West Basin and Range Province Oregon Trail Mormon Corridor Calumet Region Southwest Old Southwest Llano Estacado Central United States Tallgrass
prairie South South Central Deep South Upland South Four Corners East Coast West Coast Gulf Coast Third Coast Coastal states Eastern United States Appalachia
Trans - Mississippi Great North Woods Great Plains Interior Plains Great Lakes Great Basin Great Basin Desert Acadia Ozarks Ark - La - Tex
</item 1>

<item 2>
Dead Sea
Northwest British Isles English Channel Channel Islands Cotentin Peninsula Normandy Brittany Gulf of Lion Iberia Al - Andalus Baetic System Pyrenees Alpide belt
South Italian Peninsula Insular Italy Tuscan Archipelago Aegadian Islands Iberia Al - Andalus Baetic System Gibraltar Arc Southeastern Mediterranean Crimea
Alpide belt Germanic Celtic Slavic countries Uralic European Plain Eurasian Steppe Pontic -- Caspian steppe Wild Fields Pannonian Basin Great Hungarian Plain
Little Hungarian Plain Eastern Slovak Lowland Regions of North America Northern Eastern Canada Western Canada Canadian Prairies Central Canada Northern Canada
Atlantic Canada The Maritimes French Canada English Canada Acadia Acadian Peninsula Quebec City -- Windsor Corridor Peace River Country Cypress Hills Palliser
's Triangle Canadian Shield Interior Alaska - Yukon lowland taiga Newfoundland ( island ) Vancouver Island Gulf Islands Strait of Georgia Canadian Arctic
Archipelago Labrador Peninsula Gaspé Peninsula Avalon Peninsula Bay de Verde Peninsula Brodeur Peninsula Melville Peninsula Bruce Peninsula Banks Peninsula (
Nunavut ) Cook Peninsula Gulf of Boothia Georgian Bay Hudson Bay James Bay Greenland Pacific Northwest Inland Northwest Northeast New England Mid-Atlantic
Commonwealth West Midwest Upper Midwest Mountain States Intermountain West Basin and Range Province Oregon Trail
</item 2>

<item 3>
Geography of Prince Edward Island
The highest point of land is located at Springton in Queens County ( 152 metres or 499 feet above sea level ) . Contents ( hide ) 1 Communities 2 Coastline 3
See also 4 References 5 External links Communities ( edit ) The smaller rural communities as well as the towns and villages throughout the province proudly
retain a slower - paced , old world flavor , something that factors heavily into Prince Edward Island 's popularity as a destination for relaxation . Most rural
communities on Prince Edward Island are based on small - scale agriculture , given that the size of farm properties is quite small when compared with other
areas in Canada . There is an increasing amount of industrial farming as older farm properties are consolidated and modernized . Coastline ( edit ) The
coastline of the island consists of a combination of long beaches , dunes , red sandstone cliffs , salt water marshes and numerous bays and harbours . The
beaches , dunes and sandstone cliffs consist of sedimentary rock and other material with a high iron concentration which oxidizes upon exposure to the air .
</item 3>

<item 4>
Geography of Prince Edward Island
See also ( edit ) Geography of Canada portal List of parks in Prince Edward Island List of islands of Prince Edward Island References ( edit ) External links (
edit ) Geography of the provinces and territories of Canada National Geography of Canada Centre of Canada Provinces British Columbia Alberta Saskatchewan
Manitoba Ontario Quebec New Brunswick Nova Scotia Prince Edward Island Newfoundland and Labrador Territories Yukon Northwest Territories Nunavut See also
Regions of Canada Islands Rivers Lakes Mountains National Parks Volcanoes Portal WikiProject Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Geography_of_Prince_Edward_Island&oldid=835998214 '' Categories : Geography of Prince Edward Island Hidden categories
: Articles with peacock terms from August 2017 All articles with peacock terms Talk Contents About Wikipedia Français Edit links This page was last edited on 12
April 2018 , at 01 : 54 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you
agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About
Wikipedia
</item 4>

<item 5>
Geography of Prince Edward Island
Geography of Prince Edward Island - wikipedia Geography of Prince Edward Island Jump to : navigation , search This article contains wording that promotes the
subject in a subjective manner without imparting real information . Please remove or replace such wording and instead of making proclamations about a subject 's
importance , use facts and attribution to demonstrate that importance . ( August 2017 ) ( Learn how and when to remove this template message ) Prince Edward
Island The coast of Prince Edward Island around Cavendish Geography Location Gulf of Saint Lawrence Area 5,683.91 km ( 2,194.57 sq mi ) Highest elevation 152 m
( 499 ft ) Administration Canada Province Prince Edward Island Largest settlement Charlottetown ( pop . 32,174 ) Demographics Population 139,407 ( 2008 ) Pop .
</item 5>

<item 6>
Geography of Prince Edward Island
( See also a list of communities in Prince Edward Island . ) The island 's landscape is pastoral : rolling hills , pristine forests , white sand beaches , ocean
coves and the famous red soil have given Prince Edward Island a reputation as a province of outstanding natural beauty . A number of laws have been passed by
the provincial government to attempt to preserve the landscape through regulation , although the lack of consistent enforcement and absence of province - wide
zoning and land - use planning has resulted in some aesthetically displeasing development in recent years . The island 's lush landscape has had a strong
bearing not only on its economy but also its culture . Author Lucy Maud Montgomery drew inspiration from the land during the late Victorian Era for the setting
of her classic novel Anne of Green Gables . Today , many of the same qualities that Montgomery and others found in the Island are enjoyed by tourists who visit
during all seasons . They enjoy a variety of leisure activities , including beaches , various golf courses , eco-tourism adventures , and simply touring the
countryside and enjoying cultural events in local communities around the island .
</item 6>

<item 7>
Anne (TV series)
CBC News . Retrieved 4 June 2017 . ^ Jump up to : Ahearn , Victoria ( March 16 , 2017 ) . `` VIDEO : CBC 's ' Anne ' shows darker past of ' accidental feminist
' from Green Gables '' . Chronicle Herald . Canadian Press . Retrieved 4 June 2017 . Jump up ^ `` CBC - TV and Netflix renew Anne series for 2nd season :
Episodes will be written by team of female writers '' . CBC News Prince Edward Island . Canadian Press . 3 August 2017 . Retrieved 5 August 2017 . Jump up ^
Anderson , Lance ( 8 November 2016 ) . `` New series based on Anne of Green Gables filmed in Millbrook on Tuesday - MyKawartha.com '' . Jump up ^ Fraser , Sara
( November 28 , 2016 ) . `` Behind the scenes on the newest Anne of Green Gables series '' . CBC News . Retrieved April 4 , 2017 . Jump up ^ `` Get Red - dy !
</item 7>

<item 8>
Geography of Prince Edward Island
The geological properties of a white silica sand found at Basin Head are unique in the province ; the sand grains cause a scrubbing noise as they rub against
each other when walked on . Large dune fields on the north shore can be found on barrier islands at the entrances to various bays and harbours . The magnificent
sand dunes at Greenwich are of particular significance . The shifting , parabolic dune system is home to a variety of birds and rare plants and is also a site
of significant archeological interest .
</item 8>

<item 9>
Geography of Prince Edward Island
32,174 ) Demographics Population 139,407 ( 2008 ) Pop . density 24.53 / km ( 63.53 / sq mi ) Ethnic groups Scottish ( 38.0 % ) , English ( 28.7 % ) , Irish (
27.9 % ) , French ( 21.3 % ) , German ( 4.0 % ) , and Dutch ( 3.1 % ) Prince Edward Island 's geography is mostly pastoral with red soil , white sand , and
scattered communities . Known as the `` Garden of the Gulf '' , the island is located in the Gulf of Saint Lawrence north of Nova Scotia and east of New
Brunswick , with which it forms the Northumberland Strait . PEI consists of the capital city Charlottetown , as well as urban towns Cornwall and Stratford and a
developing urban fringe . A smaller urban area surrounds Summerside Harbor , situated on the southern shore 40 kilometers west of Charlottetown Harbor , and
consists primarily of the city of Summerside . As with all natural harbors on the island , Charlottetown and Summerside harbors are created by rias .
</item 9>

<item 10>
Governor-General of India
Governors - General of India ( 1833 -- 1858 ) Bentinck Metcalfe Auckland Ellenborough Bird Hardinge Dalhousie Canning Governors - General and Viceroys of India
( 1858 -- 1947 ) Canning Elgin Napier Denison Lawrence Mayo Strachey Napier Northbrook Lytton Ripon Dufferin Lansdowne Elgin Curzon Minto Penshurst Chelmsford
Reading Irwin Willingdon Wavell Mountbatten Governors - General of the Union of India ( 1947 -- 1950 ) Mountbatten Rajagopalachari Representatives of the
monarch in Commonwealth realms and Dominions National Counsellors of State United Kingdom Governors - general Antigua and Barbuda Australia Bahamas Barbados
Belize Canada Grenada Jamaica New Zealand Papua New Guinea Saint Kitts and Nevis Saint Lucia Saint Vincent and the Grenadines Solomon Islands Tuvalu
Sub-national Australia Governors New South Wales Queensland South Australia Tasmania Victoria Western Australia Administrator Northern Territory Canada
Lieutenant governors Alberta British Columbia Manitoba New Brunswick Newfoundland Nova Scotia Ontario Prince Edward Island Quebec Saskatchewan Realm of New
Zealand Governor - general Niue , New Zealand Queen 's Representative Cook Islands Administrator Tokelau United Kingdom Lords lieutenant England ,
</item 10>
=== END ITEM id=c8bacd8f0ff3f207 ===

=== ITEM id=6df1bb6a60f2668b ===
Gold sentences and facts:
Sentence 0: `` Go Tell It on the Mountain '' is an African - American spiritual song , compiled by John Wesley Work , Jr. , dating back to at least 1865 , that has been sung and recorded by many gospel and secular performers . It is considered a Christmas carol because its original lyrics celebrate the Nativity of Jesus :
  Fact 0.0: 'Go Tell It on the Mountain' is an African-American spiritual song.
  Fact 0.1: 'Go Tell It on the Mountain' was compiled by John Wesley Work, Jr.
  Fact 0.2: 'Go Tell It on the Mountain' dates back to at least 1865.
  Fact 0.3: 'Go Tell It on the Mountain' has been sung and recorded by many gospel performers.
  Fact 0.4: 'Go Tell It on the Mountain' has been sung and recorded by many secular performers.
  Fact 0.5: 'Go Tell It on the Mountain' is considered a Christmas carol.
  Fact 0.6: The original lyrics of 'Go Tell It on the Mountain' celebrate the Nativity of Jesus.
  Fact 0.7: 'Go Tell It on the Mountain' is considered a Christmas carol because its original lyrics celebrate the Nativity of Jesus.

Retrieved items:
<item 1>
Go Tell It on the Mountain (song)
on the album ( various artists ) 2003 : The Blind Boys of Alabama on the album Go Tell It on the Mountain 2004 : James Taylor on the album A Christmas Album
2007 : Toby Keith on the album A Classic Christmas 2008 : Needtobreathe on the album An Alternative Christmas ( various artists ) 2008 : Sheryl Crow on the
album Home for Christmas 2009 : Big Daddy Weave on the album Christ Is Come 2012 : Pentatonix on the album PTXmas 2013 : David Crowder Band on the album Wow
Christmas : 30 Top Christian Artists and Holiday Songs 2015 : Hunter Hayes on the album Country Faith Christmas ( various artists ) 2016 : Jennifer Nettles on
the album To Celebrate Christmas 2016 : Sarah McLachlan on the album Wonderland 2016 : Blues Traveler on Go Tell It On The Mountain ( single ) 2017 : Hanson (
band ) - mashup of ' Joy To The World ' and ' Go Tell It On The Mountain ' ( ' Joy To The Mountain ' ) on the album Finally It 's Christmas References ( edit )
Jump up ^ Marsh ,
</item 1>

<item 2>
Go Tell It on the Mountain (song)
Unsourced material may be challenged and removed . ( December 2016 ) ( Learn how and when to remove this template message ) In 1963 , the musical team Peter ,
Paul and Mary , along with their musical director Milt Okun , adapted and rewrote `` Go Tell It on the Mountain '' as `` Tell It on the Mountain '' , their
lyrics referring specifically to : Exodus and using the phrase `` Let my people go , '' but referring implicitly to the Civil Rights struggle of the early 1960s
. This version became a moderately successful single for them ( US # 33 pop , 1964 ) . According to Religious Studies professor and Civil Rights historian
Charles Marsh , it was African American Civil Rights leader Fannie Lou Hamer who combined this song with the spiritual `` Go Down Moses , '' taking the last
line of the chorus , `` Let my people go '' and substituting it in the chorus of `` Go Tell It on the Mountain . ''
</item 2>

<item 3>
Go Tell It on the Mountain (song)
Christmas Christmas Eve Children 's Day Boxing Day Nochebuena Saint Nicholas Day St. Stephen 's Day Sol Invictus Yule In Christianity Biblical Magi Adoration of
the Magi Adoration of the Shepherds Advent Angel Gabriel Annunciation Annunciation to the shepherds Baptism of the Lord Bethlehem Christingle Christmastide
Epiphany Herod the Great Jesus Joseph Mary Massacre of the Innocents flight into Egypt Nativity Fast Nativity of Jesus in art in later culture Nativity scene
Saint Nicholas Star of Bethlehem Twelfth Night In folklore Badalisc La Befana Belsnickel Caganer Christkind Ded Moroz Elves Father Christmas Grýla Jack Frost
Joulupukki Knecht Ruprecht Korvatunturi Krampus Mikulás Miner 's figure Mrs. Claus Nisse / Tomte North Pole Old Man Winter Olentzero Père Fouettard Père Noël
Rudolph the Red - Nosed Reindeer Saint Lucy Santa 's reindeer Santa 's workshop Sinterklaas Tió de Nadal Vertep Yule Cat Yule Lads Zwarte Piet Gift - bringers
Saint Nicholas Santa Claus List of
</item 3>

<item 4>
Go Tell It on the Mountain (song)
3 AM 1965 : Ray Conniff Singers on the album Here We Come A-Caroling 1971 : Bob Marley and the Wailers featuring Peter Tosh on the album The Best of The Wailers
1981 : Anne Murray on the album Christmas Wishes 1990 : Dolly Parton on the album Home for Christmas 1993 : Bruce Cockburn on the album Christmas 1996 : Vanessa
L. Williams on the album Star Bright 1998 : Cece Winans on the album His Gift 1999 : Garth Brooks on the album Garth Brooks and the Magic of Christmas 1999 :
Jewel on the album Joy : A Holiday Collection 2001 : Candi Staton on the album Christmas in My Heart 2001 : Fred Hammond on the album Christmas . Just Remember
2001 : Sara Evans on the album Country Christmas 2001 2002 : Jim Nabors on the album Jim Nabors Christmas 2003 : Oh Susanna on the album Maybe This Christmas
Too ?
</item 4>

<item 5>
Where Have All the Flowers Gone?
Versions ( edit ) The Kingston Trio recorded the song in 1961 . Believing it to be a traditional song , they claimed authorship , although upon notice from
Seeger they had their name removed and credited Seeger and Hickerson . Seeger acknowledged their success with this song . Their single , with `` O Ken Karanga
'' as the A-side and the hit `` Where Have All The Flowers Gone ? '' the B - side , reached # 21 in the 1962 Billboard Hot 100 chart and # 4 on the Easy
Listening chart . Peter , Paul and Mary included the song on their eponymous debut album ( which spent five weeks as the # 1 album in the country ) in 1962 .
Marlene Dietrich performed the song in English , French , and German . The song was first performed in French ( as `` Qui peut dire où vont les fleurs ? '' ) by
Marlene in 1962 at a UNICEF concert . She also recorded the song in English and in German , the latter titled `` Sag ' mir , wo die Blumen sind '' , with lyrics
translated by Max Colpet .
</item 5>

<item 6>
Go Tell It on the Mountain (song)
Dumpling Hallaca Tamale Meat Ham Roast goose Romeritos Turkey Stuffing Portal Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Go_Tell_It_on_the_Mountain_(song)&oldid=812507828 '' Categories : Christmas carols Songs about mountains Songs about
Jesus Mahalia Jackson songs Peter , Paul and Mary songs Hidden categories : Articles needing additional references from December 2016 All articles needing
additional references Talk About Wikipedia Deutsch Italiano Svenska Tagalog Edit links This page was last edited on 28 November 2017 , at 06 : 07 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 6>

<item 7>
Where Have All the Flowers Gone?
Pete Seeger , The Kingston Trio , Peter , Paul and Mary , Johnny Rivers , many others Esperanto Ĉiuj floroj estas for Duo Espera Estonian Kuhu küll kõik lilled
jäid Lindpriid Finnish Minne kukat kadonneet Kukonpojat French Qui peut dire où vont les fleurs ? Eva ( de ) , Marlene Dietrich , Francis Lemarque , Dalida
German Sag mir , wo die Blumen sind Marlene Dietrich , Hannes Wader , Juliane Werding , Nana Mouskouri , Joan Baez , Lolita ( Austrian singer ) , Hildegard Knef
, Einstürzende Neubauten Hebrew איפה הפרחים כולם Eifo Haprachim Kulam שלישיית גשר הירקון ( Gesher Hayarkon Trio ) Hungarian Hova tűnt a sok virág ? Mária Mezei
, Péter Gerendás Icelandic Hvert er farið blómið blátt ?
</item 7>

<item 8>
Go Tell It on the Mountain (song)
Marsh does not document this claim , but notes that Hamer was highly active in civil rights work beginning in the 1950s , and that the use of the Exodus story
and the singing of spirituals played a central role in her activities . Little Big Town 's 2006 non-album single version reached No. 35 on the Hot Country Songs
chart . Other artists who have recorded the song ( chiefly on either Christmas - themed music albums or collections of spirituals or folk songs ) include : 195
? : Mahalia Jackson on the album Silent Night , Holy Night 1960 : Bobby Darin on the album The 25th Day of December 1964 : Gordon MacRae on the album Firestone
Presents Your Christmas Favorites , Volume 3 1964 : Bing Crosby and Frank Sinatra on the album 12 Songs of Christmas 1964 : Simon & Garfunkel on the album
Wednesday Morning ,
</item 8>

<item 9>
Go Tell It on the Mountain (song)
Charles ( 1997 ) . God 's Long Summer . Princeton . p. 47 . ISBN 9780691130675 . Jump up ^ Whitburn , Joel ( 2008 ) . Hot Country Songs 1944 to 2008 . Record
Research , Inc. p. 241 . ISBN 0 - 89820 - 177 - 2 . Jump up ^ `` Firestone Your Favorite Christmas Music - Volume Three - 1964 '' . DLF Music Transfer. 2006 .
Christmas Christmas Eve Children 's Day Boxing Day Nochebuena Saint Nicholas Day St.
</item 9>

<item 10>
Go Tell It on the Mountain (song)
Go Tell It on the Mountain ( song ) - wikipedia Go Tell It on the Mountain ( song ) Jump to : navigation , search Go Tell It on the Mountain by unknown Genre
Christmas , spiritual Language English Published 1909 `` Go Tell It on the Mountain '' is an African - American spiritual song , compiled by John Wesley Work ,
Jr. , dating back to at least 1865 , that has been sung and recorded by many gospel and secular performers . It is considered a Christmas carol because its
original lyrics celebrate the Nativity of Jesus : `` Go tell it on the mountain , over the hills and everywhere ; go tell it on the mountain , that Jesus Christ
is born . '' Recording artists ( edit ) This section needs additional citations for verification . Please help improve this article by adding citations to
reliable sources . Unsourced material may be challenged and removed .
</item 10>
=== END ITEM id=6df1bb6a60f2668b ===

=== ITEM id=6b420ef9d54f5ece ===
Gold sentences and facts:
Sentence 0: Rank Team Won Lost Tied Pct . First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744 568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684 572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens 181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401 11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400 460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392 468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32 Tampa Bay Buccaneers 250 393 . 389 1976 644 NFC South
  Fact 0.0: The table columns are rank, team, won, lost, tied, pct, first NFL season, total games, and division.
  Fact 0.1: The Dallas Cowboys won 493 games, lost 367, tied 6, had a pct of .573, had a first NFL season of 1960, played 866 total games, and are in the NFC East.
  Fact 0.2: The Green Bay Packers won 730 games, lost 553, tied 37, had a pct of .567, had a first NFL season of 1921, played 1,320 total games, and are in the NFC North.
  Fact 0.3: The Chicago Bears won 744 games, lost 568, tied 42, had a pct of .565, had a first NFL season of 1920, played 1,354 total games, and are in the NFC North.
  Fact 0.4: The Miami Dolphins won 439 games, lost 341, had a pct of .563, had a first NFL season of 1966, played 784 total games, and are in the AFC East.
  Fact 0.5: Rank 5 is the New England Patriots, who won 476 games, lost 383, tied 9, had a pct of .554, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.6: Rank 6 is the New York Giants, who won 684 games, lost 572, tied 33, had a pct of .543, had a first NFL season of 1925, played 1,289 total games, and are in the NFC East.
  Fact 0.7: Rank 7 is the Denver Broncos, who won 465 games, lost 393, tied 10, had a pct of .541, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.8: Rank 8 is the Minnesota Vikings, who won 457 games, lost 387, tied 10, had a pct of .541, had a first NFL season of 1961, played 854 total games, and are in the NFC North.
  Fact 0.9: Rank 9 is the Baltimore Ravens, who won 181 games, lost 154, had a pct of .540, played 336 total games, and are in the AFC North.
  Fact 0.10: Rank 10 is the San Francisco 49ers, who won 522 games, lost 450, tied 14, had a pct of .537, had a first NFL season of 1950, played 986 total games, and are in the NFC West.
  Fact 0.11: Rank 11 is the Indianapolis Colts, who won 502 games, lost 441, tied 7, had a pct of .532, had a first NFL season of 1953, played 950 total games, and are in the AFC South.
  Fact 0.12: Rank 12 is the Oakland Raiders, who won 456 games, lost 401, tied 11, had a pct of .532, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.13: Rank 13 is the Pittsburgh Steelers, who won 601 games, lost 535, tied 20, had a pct of .529, had a first NFL season of 1933, played 1,156 total games, and are in the AFC North.
  Fact 0.14: Rank 14 is the Kansas City Chiefs, who won 447 games, lost 409, tied 12, had a pct of .522, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.15: Rank 15 is the Washington Redskins, who won 586 games, lost 572, tied 28, had a pct of .506, had a first NFL season of 1932, played 1,186 total games, and are in the NFC East.
  Fact 0.16: Rank 16 is the Seattle Seahawks, who won 325 games, lost 318, had a pct of .505, had a first NFL season of 1976, played 644 total games, and are in the NFC West.
  Fact 0.17: Rank 17 is the Cleveland Browns, who won 462 games, lost 466, tied 10, had a pct of .498, had a first NFL season of 1950, played 938 total games, and are in the AFC North.
  Fact 0.18: Rank 18 is the Los Angeles Chargers, who won 426 games, lost 431, tied 11, had a pct of .497, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.19: Rank 19 is the Los Angeles Rams, who won 544 games, lost 554, tied 21, had a pct of .496, had a first NFL season of 1937, played 1,119 total games, and are in the NFC West.
  Fact 0.20: Rank 20 is the Carolina Panthers, who won 172 games, lost 179, had a pct of .490, had a first NFL season of 1995, played 352 total games, and are in the NFC South.
  Fact 0.21: Rank 21 is the Philadelphia Eagles, who won 555 games, lost 591, tied 26, had a pct of .485, had a first NFL season of 1933, played 1,172 total games, and are in the NFC East.
  Fact 0.22: Rank 22 is the Tennessee Titans, who won 413 games, lost 449, tied 6, had a pct of .479, had a first NFL season of 1960, played 868 total games, and are in the AFC South.
  Fact 0.23: Rank 23 is the Buffalo Bills, who won 400 games, lost 460, tied 8, had a pct of .465, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.24: Rank 24 is the Detroit Lions, who won 544 games, lost 641, tied 32, had a pct of .460, had a first NFL season of 1930, played 1,217 total games, and are in the NFC North.
  Fact 0.25: Rank 25 is the Cincinnati Bengals, who won 344 games, lost 408, had a pct of .458, had a first NFL season of 1968, played 756 total games, and are in the AFC North.
  Fact 0.26: Rank 26 is the New York Jets, who won 392 games, lost 468, tied 8, had a pct of .456, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.27: Rank 27 is the New Orleans Saints, who won 338 games, lost 427, tied 5, had a pct of .442, had a first NFL season of 1967, played 770 total games, and are in the NFC South.
  Fact 0.28: Rank 28 is the Houston Texans, who won 106 games, lost 134, tied 0, had a pct of .442, had a first NFL season of 2002, played 240 total games, and are in the AFC South.
  Fact 0.29: Rank 29 is the Jacksonville Jaguars, who won 155 games, lost 197, tied 0, had a pct of .440, had a first NFL season of 1995, played 352 total games, and are in the AFC South.
  Fact 0.30: Rank 30 is the Atlanta Falcons, who won 341 games, lost 437, tied 6, had a pct of .439, had a first NFL season of 1966, played 784 total games, and are in the NFC South.
  Fact 0.31: Rank 31 is the Arizona Cardinals, who won 542 games, lost 732, tied 40, had a pct of .428, had a first NFL season of 1920, played 1,314 total games, and are in the NFC West.
  Fact 0.32: Rank 32 is the Tampa Bay Buccaneers, who won 250 games, lost 393, had a pct of .389, had a first NFL season of 1976, played 644 total games, and are in the NFC South.

Retrieved items:
<item 1>
NFL win–loss records
Jump up ^ The Colts played under the `` Baltimore Colts '' name from 1953 -- 1983 , before relocating to Indianapolis and adpoting their current `` Indianapolis
Colts '' name , which they have played under since the 1984 season . References ( edit ) Jump up ^ `` Franchise Encyclopedias '' . Pro Football Reference .
Sports - Reference . Retrieved January 14 , 2016 . Jump up ^ Brown , Daniel ; Emmons , Mark ( November 8 , 2011 ) . `` 49ers great Joe Perry and the stats that
do n't count '' . San Jose Mercury News . Retrieved January 14 , 2016 . Jump up ^ `` New England Patriots -- Team History '' . Pro Football Hall of Fame .
Retrieved October 17 , 2017 . Jump up ^ Dewdney , Mike ( February 21 , 2012 ) . `` The Rise and Fall of the Bay State Patriots '' . Chris Creamer 's Sports
Logos . Retrieved October 17 , 2017 .
</item 1>

<item 2>
NFL win–loss records
490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400
460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392
468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville
Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32
Tampa Bay Buccaneers 250 393 .
</item 2>

<item 3>
NFL win–loss records
Ties are registered as a half win and a half loss when calculating the win - loss record . Best win -- loss record in division Rank Team Won Lost Tied Pct .
First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744
568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684
572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens
181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 .
</item 3>

<item 4>
List of National Football League records (individual)
Peyton Manning , Indianapolis Colts 1998 ; Derek Carr , Oakland Raiders , 2014 Most losses in a season , by a starting quarterback : 15 , Archie Manning , New
Orleans Saints , 1980 ; Jeff George , Indianapolis Colts , 1991 ; Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses , by a rookie starting
quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most losses in a season , by a rookie starting quarterback : 15 , Deshone Kizer , Cleveland Browns ,
2017 Most consecutive losses in a season , by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses to start a season ,
by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses by a starting quarterback : 21 , Dan Pastorini , Houston Oilers
, 1972 -- 1974 Most consecutive losses by a starting quarterback , regular season , to start a career : 15 , Deshone Kizer , Cleveland Browns , 2017 Most
consecutive road losses by a starting quarterback , regular season , to start a career : 13 ,
</item 4>

<item 5>
NFL win–loss records
540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401
11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington
Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los
Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC
South 21 Philadelphia Eagles 555 591 26 .
</item 5>

<item 6>
NFL win–loss records
NFL win -- loss records - wikipedia NFL win -- loss records Jump to : navigation , search It has been suggested that NFL standings since AFL -- NFL merger be
merged into this article . ( Discuss ) Proposed since October 2017 . The following is a list of win -- loss records for each of the 32 active National Football
League ( NFL ) teams . Contents ( hide ) 1 Regular season 2 Playoffs 3 Notes 4 References Regular season ( edit ) George Halas ( right ) of the Chicago Bears ,
who have recorded the most regular season wins . The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their regular
season win -- loss record percentage , accurate as of the end of the 2016 NFL regular season . For teams that previously played in the All - America Football
Conference ( AAFC ) , their AAFC win - loss records are not included as the NFL does not officially count AAFC statistics despite the 1950 NFL -- AAFC merger .
However , for teams that played in the American Football League ( AFL ) , their AFL win - loss records are included and officially counted by the NFL .
</item 6>

<item 7>
NFL win–loss records
480 AFC East 18 Philadelphia Eagles 40 19 21 . 475 2013 NFC East 19 Jacksonville Jaguars 11 5 6 . 455 2007 AFC South 20 Los Angeles Rams 43 19 24 . 442 NFC West
21 ( tie ) Arizona Cardinals 16 7 9 . 438 2015 NFC West 21 ( tie ) New Orleans Saints 16 7 9 . 438 2013 NFC South 23 Houston Texans 7 . 429 2016 AFC South 24
Tennessee Titans 33 14 19 . 424 2008 AFC South 25 Atlanta Falcons 22 9 13 . 409 2016 NFC South 26 Minnesota Vikings 47 19 28 . 404 2015 NFC North 27 Tampa Bay
Buccaneers 15 6 9 . 400 2007 NFC South 28 Los Angeles Chargers 28 11 17 . 393 2013 AFC West 29 Cleveland Browns 31 11 20 . 355 2002 AFC North 30 Detroit Lions
20 7 13 .
</item 7>

<item 8>
NFL win–loss records
389 1976 644 NFC South Playoffs ( edit ) Members of the Baltimore Ravens franchise , the team with the highest playoff win -- loss percentage , during the
Lombardi Trophy presentation following their Super Bowl XLVII victory Marvin Lewis , head coach of the Cincinnati Bengals , who have registered the lowest
playoff win - loss percentage of all time The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their playoff win - loss
record percentage , accurate as of the end of Super Bowl LI . Best win -- loss record in division Team active in playoffs Rank Team Games Won Lost Pct . Last
playoff appearance Division Baltimore Ravens 23 15 8 . 652 2014 AFC North New England Patriots 51 32 19 . 627 2016 AFC East Green Bay Packers 56 34 22 . 607
2016 NFC North 4 ( tie ) Pittsburgh Steelers 60 36 24 . 600 2016 AFC North 4 ( tie ) San Francisco 49ers 50 30 20 . 600 2013 NFC West 6 Oakland Raiders 44 25 19
.
</item 8>

<item 9>
NFL win–loss records
600 2013 NFC West 6 Oakland Raiders 44 25 19 . 568 2016 AFC West 7 Carolina Panthers 16 9 7 . 563 2015 NFC South 8 Dallas Cowboys 61 34 27 . 557 2016 NFC East 9
( tie ) Washington Redskins 42 23 19 . 548 2015 NFC East 9 ( tie ) Denver Broncos 42 23 19 . 548 2015 AFC West 11 Seattle Seahawks 31 16 15 . 516 2016 NFC West
12 New York Giants 49 24 25 . 490 2016 NFC East 13 Indianapolis Colts 45 22 23 . 489 2014 AFC South 14 Miami Dolphins 41 20 21 . 488 2016 AFC East 15 Chicago
Bears 35 17 18 . 486 NFC North 16 Buffalo Bills 29 14 15 . 483 1999 AFC East 17 New York Jets 25 12 13 . 480 AFC East 18 Philadelphia Eagles 40 19 21 .
</item 9>

<item 10>
NFL win–loss records
355 2002 AFC North 30 Detroit Lions 20 7 13 . 350 2016 NFC North 31 Kansas City Chiefs 26 9 17 . 346 2016 AFC West 32 Cincinnati Bengals 19 5 14 . 263 2015 AFC
North Notes ( edit ) Jump up ^ For example , the Browns ' regular season win -- loss record does not include a 47 -- 4 -- 3 record compiled in the All - America
Football Conference from 1946 to 1949 , nor does their playoff record include the 5 -- 0 record compiled during the same period . Likewise , the 49ers ' AAFC
regular season win -- loss record of 38 -- 14 -- 2 , nor their 1 - 1 AAFC playoff record are officially counted Jump up ^ The Patriots played as the `` Boston
Patriots '' from 1960 -- 1970 , before adopting the `` New England Patriots '' name , which they have played under since the 1971 season . The team was also
briefly known as the `` Bay State Patriots '' prior to the 1971 season .
</item 10>
=== END ITEM id=6b420ef9d54f5ece ===
