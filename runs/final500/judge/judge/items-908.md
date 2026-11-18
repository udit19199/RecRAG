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

=== ITEM id=f25efa0843a85375 ===
Gold sentences and facts:
Sentence 0: A primary ossification center is the first area of a bone to start ossifying . It usually appears during prenatal development in the central part of each developing bone . In long bones the primary centers occur in the diaphysis / shaft and in irregular bones the primary centers occur usually in the body of the bone . Most bones have only one primary center ( e.g. all long bones ) but some irregular bones such as the os coxa ( hip ) and vertebrae have multiple primary centers .
  Fact 0.0: A primary ossification center is the first area of a bone to start ossifying.
  Fact 0.1: A primary ossification center usually appears during prenatal development.
  Fact 0.2: A primary ossification center usually appears in the central part of each developing bone.
  Fact 0.3: In long bones, the primary centers occur in the diaphysis (shaft).
  Fact 0.4: In irregular bones, the primary centers usually occur in the body of the bone.
  Fact 0.5: Most bones have only one primary center, such as all long bones.
  Fact 0.6: Some irregular bones have multiple primary centers.
  Fact 0.7: The os coxa (hip) is an irregular bone with multiple primary centers.
  Fact 0.8: Vertebrae have multiple primary centers.

Retrieved items:
<item 1>
Ossification center
Anatomy , descriptive and applied . the University of California : Lea & Febiger . Development of bone Ossification Triradiate cartilage Limb development Limb
bud Apical ectodermal ridge Zone of polarizing activity Sclerotome Myotome Septum transversum Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Ossification_center&oldid=808049222 '' Categories : Wikipedia articles incorporating text from the 20th edition of
Gray 's Anatomy ( 1918 ) Skeletal system Tissues ( biology ) Hidden categories : Articles needing additional references from May 2011 All articles needing
additional references Medicine infobox template using GraySubject or GrayPage Medicine infobox template using Dorlands parameter Talk About Wikipedia Nederlands
Edit links This page was last edited on 31 October 2017 , at 15 : 42 . About Wikipedia
</item 1>

<item 2>
Ossification center
Ossification center - wikipedia Ossification center Jump to : navigation , search This article needs additional citations for verification . Please help improve
this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2011 ) ( Learn how and when to remove this
template message ) Ossification center X-ray of ossification centers in a child . Details Identifiers Latin centrum ossificationis Dorlands / Elsevier 12227335
TA A02. 0.00. 043 FMA 75436 Anatomical terminology ( edit on Wikidata ) The first step in ossification of the cartilage is that the cartilage cells , at the
point where ossification is commencing and which is termed as an ossification center , enlarge and arrange themselves in rows . The matrix in which they are
imbedded increases in quantity , so that the cells become further separated from each other . A deposit of calcareous material now takes place in this matrix ,
between the rows of cells , so that they become separated from each other by longitudinal columns of calcified matrix , presenting a granular and opaque
appearance .
</item 2>

<item 3>
Human fertilization
1 Cortical reaction 2 Fusion 2.1 Cell membranes 2.2 Transformations 2.3 Replication 2.4 Mitosis 3 Fertilization age 4 Diseases 5 See also 6 References 7
External links Anatomy ( edit ) Corona radiata ( edit ) The sperm binds through the corona radiata , a layer of follicle cells on the outside of the secondary
oocyte . Fertilization occurs when the nucleus of both a sperm and an egg fuse to form a diploid cell , known as zygote . The successful fusion of gametes forms
a new organism . Cone of attraction and perivitelline membrane ( edit ) Where the spermatozoon is about to pierce , the yolk ( ooplasm ) is drawn out into a
conical elevation , termed the cone of attraction or reception cone . Once the spermatozoon has entered , the peripheral portion of the yolk changes into a
membrane , the perivitelline membrane , which prevents the passage of additional spermatozoa .
</item 3>

<item 4>
Middle cranial fossa
p. 154 . ISBN 9781437735802 . Additional images ( edit ) Middle cranial fossa at human foetus Base of skull Middle cranial fossa Middle cranial fossa Middle
cranial fossa External links ( edit ) Wikimedia Commons has media related to Middle cranial fossa . Anatomy photo : 22 : os - 0802 at the SUNY Downstate Medical
Center Compound structures of skull Neurocranium Calvaria Diploë Asterion Pterion Stephanion Inion Bregma Lambda Fossae anterior middle posterior cranial cavity
Base of skull Fontanelle anterior posterior sphenoidal mastoid Facial skeleton Nasion Gonion Both dacryon zygomatic arch temporal fossa infratemporal fossa
pterygomaxillary fissure pterygopalatine fossa Foramina of the skull ( and canals , fissures , meatus , and hiatus ) Anterior cranial fossa to Orbit : ethmoidal
foramina anterior posterior to Nasal cavity : olfactory foramina ( CN - I ) foramen cecum Middle cranial fossa to Orbit : optic canal ( CN - II ) superior
orbital fissure ( CN - III ,
</item 4>

<item 5>
Ossification center
Most bones have only one primary center ( e.g. all long bones ) but some irregular bones such as the os coxa ( hip ) and vertebrae have multiple primary centers
. A secondary ossification center is the area of ossification that appears after the primary ossification center has already appeared -- most of which appear
during the postnatal and adolescent years . Most bones have more than one secondary ossification center . In long bones , the secondary centres appear in the
epiphyses . References ( edit ) This article incorporates text in the public domain from the 20th edition of Gray 's Anatomy ( 1918 ) Jump up ^ Gray and Spitzka
( 1910 ) , page 44 . Bibliography ( edit ) Gray , Henry ; Spitzka , Edward Anthony ( 1910 ) . Anatomy , descriptive and applied . the University of California :
Lea & Febiger .
</item 5>

<item 6>
Brain
Once neurons have positioned themselves , their axons sprout and navigate through the brain , branching and extending as they go , until the tips reach their
targets and form synaptic connections . In a number of parts of the nervous system , neurons and synapses are produced in excessive numbers during the early
stages , and then the unneeded ones are pruned away . For vertebrates , the early stages of neural development are similar across all species . As the embryo
transforms from a round blob of cells into a wormlike structure , a narrow strip of ectoderm running along the midline of the back is induced to become the
neural plate , the precursor of the nervous system . The neural plate folds inward to form the neural groove , and then the lips that line the groove merge to
enclose the neural tube , a hollow cord of cells with a fluid - filled ventricle at the center . At the front end , the ventricles and cord swell to form three
vesicles that are the precursors of the forebrain , midbrain , and hindbrain .
</item 6>

<item 7>
Retina
The role of Pax6 in eye development was elegantly demonstrated by Walter Gehring and colleagues , who showed that ectopic expression of Pax6 can lead to eye
formation on Drosophila antennae , wings , and legs . The optic vesicle gives rise to three structures : the neural retina , the retinal pigmented epithelium ,
and the optic stalk . The neural retina contains the retinal progenitor cells ( RPCs ) that give rise to the seven cell types of the retina . Differentiation
begins with the retinal ganglion cells and concludes with production of the Muller glia . Although each cell type differentiates from the RPCs in a sequential
order , there is considerable overlap in the timing of when individual cell types differentiate . The cues that determine a RPC daughter cell fate are coded by
multiple transcription factor families including the bHLH and homeodomain factors . In addition to guiding cell fate determination , cues exist in the retina to
determine the dorsal - ventral ( D-V ) and nasal - temporal ( N-T ) axes .
</item 7>

<item 8>
Human digestive system
It is from these ganglia that many of the gastric plexuses arise . Development ( edit ) Main article : Development of the digestive system Early in embryonic
development , the embryo has three germ layers and abuts a yolk sac . During the second week of development , the embryo grows and begins to surround and
envelop portions of this sac . The enveloped portions form the basis for the adult gastrointestinal tract . Sections of this foregut begin to differentiate into
the organs of the gastrointestinal tract , such as the oesophagus , stomach , and intestines . During the fourth week of development , the stomach rotates . The
stomach , originally lying in the midline of the embryo , rotates so that its body is on the left . This rotation also affects the part of the gastrointestinal
tube immediately below the stomach , which will go on to become the duodenum . By the end of the fourth week , the developing duodenum begins to spout a small
outpouching on its right side , the hepatic diverticulum , which will go on to become the biliary tree .
</item 8>

<item 9>
Ossification center
Here and there the matrix between two cells of the same row also becomes calcified , and transverse bars of calcified substance stretch across from one
calcareous column to another . Thus there are longitudinal groups of the cartilage cells enclosed in oblong cavities , the walls of which are formed of
calcified matrix which cuts off all nutrition from the cells ; the cells , in consequence , atrophy , leaving spaces called the primary areolæ . Types of
ossification centers ( edit ) Section of fetal bone of cat . ir . Irruption of the subperiosteal tissue . p . Fibrous layer of the periosteum . o . Layer of
osteoblasts . im . Subperiosteal bony deposit . There are two types of ossification centers -- primary and secondary . A primary ossification center is the
first area of a bone to start ossifying . It usually appears during prenatal development in the central part of each developing bone . In long bones the primary
centers occur in the diaphysis / shaft and in irregular bones the primary centers occur usually in the body of the bone .
</item 9>

<item 10>
Gallbladder
Early in development , the human embryo has three germ layers and abuts an embryonic yolk sac . During the second week of embryogenesis , as the embryo grows ,
it begins to surround and envelop portions of this sac . The enveloped portions form the basis for the adult gastrointestinal tract . Sections of this foregut
begin to differentiate into the organs of the gastrointestinal tract , such as the esophagus , stomach , and intestines . During the fourth week of
embryological development , the stomach rotates . The stomach , originally lying in the midline of the embryo , rotates so that its body is on the left . This
rotation also affects the part of the gastrointestinal tube immediately below the stomach , which will go on to become the duodenum . By the end of the fourth
week , the developing duodenum begins to spout a small outpouching on its right side , the hepatic diverticulum , which will go on to become the biliary tree .
Just below this is a second outpouching , known as the cystic diverticulum , that will eventually develop into the gallbladder . Function ( edit ) 1 .
</item 10>
=== END ITEM id=f25efa0843a85375 ===

=== ITEM id=24c22809337cc3bf ===
Gold sentences and facts:
Sentence 0: The optic chiasm or optic chiasma ( / ɒptɪk kaɪæzəm / ; Greek χίασμα , `` crossing '' , from the Greek χιάζω ' to mark with an X ' , after the Greek letter ' Χ ' , chi ) is the part of the brain where the optic nerves partially cross . The optic chiasm is located at the bottom of the brain immediately below the hypothalamus . The optic chiasm is found in all vertebrates , although in cyclostomes ( lampreys and hagfishes ) it is located within the brain .
  Fact 0.0: The optic chiasm is also called the optic chiasma.
  Fact 0.1: The name optic chiasm comes from the Greek word chiasma, meaning crossing.
  Fact 0.2: The name optic chiasm comes from the Greek word chiazo, meaning to mark with an X.
  Fact 0.3: The name optic chiasm is after the Greek letter chi.
  Fact 0.4: The optic chiasm is the part of the brain where the optic nerves partially cross.
  Fact 0.5: The optic chiasm is located at the bottom of the brain.
  Fact 0.6: The optic chiasm is located immediately below the hypothalamus.
  Fact 0.7: The optic chiasm is found in all vertebrates.
  Fact 0.8: In cyclostomes (lampreys and hagfishes), the optic chiasm is located within the brain.

Retrieved items:
<item 1>
Middle cranial fossa
It transmits to the orbital cavity the oculomotor , the trochlear , the ophthalmic division of the trigeminal , and the abducent nerves , some filaments from
the cavernous plexus of the sympathetic , and the orbital branch of the middle meningeal artery ; and from the orbital cavity a recurrent branch from the
lacrimal artery to the dura mater , and the ophthalmic veins . Behind the medial end of the superior orbital fissure is the foramen rotundum , for the passage
of the maxillary nerve . Behind and lateral to the foramen rotundum is the foramen ovale , which transmits the mandibular nerve , the accessory meningeal artery
, and the lesser superficial petrosal nerve . Medial to the foramen ovale is the foramen Vesalii , which varies in size in different individuals , and is often
absent ; when present , it opens below at the lateral side of the scaphoid fossa , and transmits a small vein .
</item 1>

<item 2>
Optic chiasm
arXiv : 1003.1872 . doi : 10.1163 / 157075611X617102 . ISSN 1570 - 7555 . Jump up ^ Purves , Dale ; Augustine , George ; Fitzpatrick , David ; Hall , William ;
LaMantia , Anthony - Samuel ; White , Leonard ( 2012 ) . Neuroscience . Sinauer Associates . p. 261 . ISBN 978 - 0 - 87893 - 695 - 3 . Jump up ^ `` eye , human
. '' Encyclopædia Britannica from Encyclopædia Britannica 2006 Ultimate Reference Suite DVD 2009 Jump up ^ Schmolesky MT , Wang Y , Creel DJ , Leventhal AG . ``
Abnormal retinotopic organization of the dorsal lateral geniculate nucleus of the tyrosinase - negative albino cat '' . J Comp Neurol. 427 ( 2 ) : 209 -- 19 .
</item 2>

<item 3>
Optic chiasm
PMID 4822986 . Jump up ^ Davis , Matthew C. ; Griessenauer , Christoph J. ; Bosmia , Anand N. ; Tubbs , R. Shane ; Shoja , Mohammadali M. ( 2014 - 01 - 01 ) .
`` The naming of the cranial nerves : A historical review '' . Clinical Anatomy. 27 ( 1 ) : 14 -- 19 . doi : 10.1002 / ca. 22345 . ISSN 1098 - 2353 . Jeffery G
( October 2001 ) . `` Architecture of the optic chiasm and the mechanisms that sculpt its development '' . Physiol. Rev. 81 ( 4 ) : 1393 -- 414 . PMID 11581492
. External links ( edit ) `` Anatomy diagram : 13048.000 - 1 '' . Roche Lexicon - illustrated navigator . Elsevier . Archived from the original on 2014 - 01 -
01 . ( hide ) The cranial nerves terminal Nuclei Septal nuclei Course no significant branches olfactory Nuclei
</item 3>

<item 4>
Optic chiasm
The cranial nerves terminal Nuclei Septal nuclei Course no significant branches olfactory Nuclei Anterior olfactory nucleus Course olfactory bulb olfactory
tract optic Nuclei Lateral geniculate nucleus Course optic chiasm optic tract oculomotor Nuclei Oculomotor nucleus Edinger -- Westphal nucleus Branches superior
parasympathetic root of ciliary ganglion / ciliary ganglion inferior trochlear Nucleus Branches no significant branches trigeminal Nuclei PSN Spinal trigeminal
nucleus MN TMN Course trigeminal ganglion Branches ophthalmic maxillary mandibular abducens Nucleus Branches no significant branches facial near origin
intermediate nerve geniculate inside facial canal greater petrosal pterygopalatine ganglion nerve to the stapedius chorda tympani lingual nerve submandibular
ganglion at stylomastoid foramen posterior auricular suprahyoid digastric stylohyoid parotid plexus temporal zygomatic buccal mandibular cervical Nuclei Facial
motor nucleus Solitary nucleus Superior salivary nucleus vestibulocochlear Nuclei Vestibular nuclei Cochlear
</item 4>

<item 5>
Optic chiasm
Other animals ( edit ) In Siamese cats with certain genotypes of the albino gene , this wiring is disrupted , with more of the nerve - crossing than is normal ,
as a number of scholars have reported . To compensate for lack of crossing in their brains , they cross their eyes ( strabismus ) . This is also seen in albino
tigers , as Guillery & Kaas report . Additional images ( edit ) Scheme showing central connections of the optic nerves and optic tracts . Base of brain . 3D
schematic representation of optic tracts Transformations of the visual field toward the visual map on the primary visual cortex . Human brainstem anterior view
Optic chiasm Optic chiasma Cerebrum. Inferior view. Deep dissection . Cerebrum. Inferior view . Deep dissection . History ( edit ) The crossing of nerve fibres
, and the impact on vision that this had , was probably first identified by Persian physician `` Esmail Jorjani '' , who appears to be Zayn al - Din Gorgani (
1042 -- 1137 ) .
</item 5>

<item 6>
Optic chiasm
For example , the right visual cortex receives the temporal visual field from the left eye , and the nasal visual field from the right eye , which results in
the right visual cortex producing a binocular image of the left hemispheric visual field . The net result of optic nerve crossing over at the optic chiasm is
for the right cerebral hemisphere to sense and process left hemispheric vision , and for the left cerebral hemisphere to sense and process right hemispheric
vision . This crossing is an adaptive feature of frontally oriented eyes , found mostly in predatory animals requiring precise visual depth perception . ( Prey
animals , with laterally positioned eyes , have little binocular vision , so there is a more complete crossover of visual signals . ) Beyond the optic chiasm ,
with crossed and uncrossed fibers , the optic nerves become optic tracts . The signals are passed on to the lateral geniculate body , in turn giving them to the
occipital cortex ( the outer matter of the rear brain ) .
</item 6>

<item 7>
Optic chiasm
J Comp Neurol. 427 ( 2 ) : 209 -- 19 . doi : 10.1002 / 1096 - 9861 ( 20001113 ) 427 : 2 < 209 : : aid - cne4 > 3.0.co ; 2 - 3 . PMID 11054689 . Jump up ^
Guillery , RW ; Kaas , JH ( June 1973 ) . `` Genetic abnormality of the visual pathways in a `` white '' tiger `` . Science . 180 ( 4092 ) : 1287 -- 9 . Bibcode
: 1973Sci ... 180.1287 G . doi : 10.1126 / science. 180.4092. 1287 . PMID 4707916 . Jump up ^ Guillery RW ( May 1974 ) . `` Visual pathways in albinos '' . Sci.
Am. 230 ( 5 ) : 44 -- 54 . doi : 10.1038 / scientificamerican0574 - 44 . PMID 4822986 .
</item 7>

<item 8>
Optic chiasm
The optic chiasm is located at the bottom of the brain immediately below the hypothalamus . The optic chiasm is found in all vertebrates , although in
cyclostomes ( lampreys and hagfishes ) it is located within the brain . Contents ( hide ) 1 Structure 2 Other animals 3 Additional images 4 History 5 See also 6
References 7 External links Structure ( edit ) The optic nerve fibres on the nasal sides of each retina cross over ( decussate ) to the opposite side of the
brain via the optic nerve at the optic chiasm ( decussation of medial fibers ) . The temporal hemiretina , on the other hand , stays on the same side . The
inferonasal retina are related to anterior portion of the optic chiasm whereas superonasal retinal fibers are related to the posterior portion of the optic
chiasm . The crossing over of optic nerve fibres at the optic chiasm allows the visual cortex to receive the same hemispheric visual field from both eyes .
Superimposing and processing these monocular visual signals allow the visual cortex to generate binocular and stereoscopic vision .
</item 8>

<item 9>
Optic chiasm
Optic chiasm - wikipedia Optic chiasm Jump to : navigation , search For other meanings of `` chiasm '' , see Chiasm . Optic chiasma Brain viewed from below ;
the front of the brain is above . Visual pathway with optic chiasm ( X shape ) is shown in red ( 1543 image from Andreas Vesalius ' Fabrica ) Optic nerves ,
chiasm , and optic tracts Details Part of Visual system Identifiers Latin chiasma opticum MeSH A08. 800.800. 120.680. 600 NeuroLex ID Optic chiasm TA A14. 1.08.
403 FMA 62045 Anatomical terms of neuroanatomy ( edit on Wikidata ) The optic chiasm or optic chiasma ( / ɒptɪk kaɪæzəm / ; Greek χίασμα , `` crossing '' , from
the Greek χιάζω ' to mark with an X ' , after the Greek letter ' Χ ' , chi ) is the part of the brain where the optic nerves partially cross .
</item 9>

<item 10>
Optic chiasm
See also ( edit ) Wikimedia Commons has media related to Optic chiasm . Chiasmal syndrome References ( edit ) Jump up ^ Colman , Andrew M. ( 2006 ) . Oxford
Dictionary of Psychology ( 2nd ed . ) . Oxford University Press . p. 530 . ISBN 0 - 19 - 861035 - 1 . Jump up ^ Bainbridge , David ( 30 June 2009 ) . Beyond the
Zonules of Zinn : A Fantastic Journey Through Your Brain . Harvard University Press . p. 162 . ISBN 978 - 0 - 674 - 02042 - 9 . Retrieved 22 November 2015 .
Jump up ^ de Lussanet , Marc H.E. ; Osse , Jan W.M. ( 2012 ) . `` An ancestral axial twist explains the contralateral forebrain and the optic chiasm in
vertebrates '' . Animal Biology . 62 ( 2 ) : 193 -- 216 . arXiv : 1003.1872 .
</item 10>
=== END ITEM id=24c22809337cc3bf ===

=== ITEM id=e5b3029a32ae3df7 ===
Gold sentences and facts:
Sentence 0: No . Title Featured guest Original release date `` It 's a Whole New Ball Game Now '' Barack Obama January 12 , 2018 ( 2018 - 01 - 12 ) Cold open : Letterman interviewing Obama on Late Show with David Letterman . On - location segment : Walking with Congressman John Lewis across the Edmund Pettus Bridge , site of the civil rights movement 's Bloody Sunday in 1965 . `` You Be the Newsman , I 'll Be Liz Taylor '' George Clooney February 9 , 2018 ( 2018 - 02 - 09 ) Cold open : Letterman and Clooney outside of LAX watching planes land and eating fast food . On - location segment : At the home of Nick Clooney meeting Iraqi refugee Hazim Avdal whom the Clooney family is sponsoring . `` You Know , She Has a Nobel Peace Prize '' Malala Yousafzai March 9 , 2018 ( 2018 - 03 - 09 ) Cold open : Letterman accompanies Yousafzai as she leads a tour group at Lady Margaret Hall , Oxford . On - location segment : Yousafzai continues to lead the tour group as Letterman tags along , the pair browse in the University of Oxford shop , and Letterman eats lunch with Yousafzai 's father . `` I Had a Paper Route Too '' Jay - Z April 6 , 2018 ( 2018 - 04 - 06 ) Cold open : None On - location segment : Letterman visits Shangri La Studio , speaks with music producer Rick Rubin and listens to Madison Ryann Ward recording . 5 `` TBA '' Tina Fey May 4 , 2018 6 `` TBA '' Howard Stern May 31 , 2018
  Fact 0.0: The table lists episode 1, titled 'It's a Whole New Ball Game Now', with featured guest Barack Obama, original release date January 12, 2018.
  Fact 0.1: The cold open for episode 1 has Letterman interviewing Obama on Late Show with David Letterman.
  Fact 0.2: The on-location segment for episode 1 has Letterman walking with Congressman John Lewis across the Edmund Pettus Bridge, site of the civil rights movement's Bloody Sunday in 1965.
  Fact 0.3: The table lists episode 2, titled 'You Be the Newsman, I'll Be Liz Taylor', with featured guest George Clooney, original release date February 9, 2018.
  Fact 0.4: The cold open for episode 2 has Letterman and Clooney outside of LAX watching planes land and eating fast food.
  Fact 0.5: The on-location segment for episode 2 is at the home of Nick Clooney meeting Iraqi refugee Hazim Avdal, whom the Clooney family is sponsoring.
  Fact 0.6: The table lists episode 3, titled 'You Know, She Has a Nobel Peace Prize', with featured guest Malala Yousafzai, original release date March 9, 2018.
  Fact 0.7: The cold open for episode 3 has Letterman accompanying Yousafzai as she leads a tour group at Lady Margaret Hall, Oxford.
  Fact 0.8: The on-location segment for episode 3 has Yousafzai continuing to lead the tour group while Letterman tags along.
  Fact 0.9: In the episode 3 on-location segment, the pair browse in the University of Oxford shop.
  Fact 0.10: In the episode 3 on-location segment, Letterman eats lunch with Yousafzai's father.
  Fact 0.11: The table lists episode 4, titled 'I Had a Paper Route Too', with featured guest Jay-Z, original release date April 6, 2018.
  Fact 0.12: The cold open for episode 4 is none.
  Fact 0.13: The on-location segment for episode 4 has Letterman visiting Shangri La Studio.
  Fact 0.14: In the episode 4 on-location segment, Letterman speaks with music producer Rick Rubin.
  Fact 0.15: In the episode 4 on-location segment, Letterman listens to Madison Ryann Ward recording.
  Fact 0.16: The table lists episode 5, titled 'TBA', with featured guest Tina Fey, original release date May 4, 2018.
  Fact 0.17: The table lists episode 6, titled 'TBA', with featured guest Howard Stern, original release date May 31, 2018.

Retrieved items:
<item 1>
My Next Guest Needs No Introduction with David Letterman
On January 5 , 2018 , Netflix released further information about the series . It was announced that Letterman 's guests in the first season would include Barack
Obama , George Clooney , Malala Yousafzai , Jay - Z , Tina Fey , and Howard Stern . It was also announced that the series would premiere on January 12 , 2018
with Letterman 's interview with Obama . This interview was Obama 's first since leaving office . Music ( edit ) The show 's theme song and bridging music were
written and recorded by Letterman 's long - time bandleader , Paul Shaffer . Soon after it was announced publicly that Netflix had given the series an order ,
Shaffer received a phone call from Letterman asking him to work on the show . Soon after , Shaffer began to receive cuts of episodes from the first season and
he started to put music in afterwards where the director thought it was needed . In developing the sound of the show 's music , Shaffer initially looked to
Letterman for guidance .
</item 1>

<item 2>
My Next Guest Needs No Introduction with David Letterman
`` I Had a Paper Route Too '' Jay - Z April 6 , 2018 ( 2018 - 04 - 06 ) Cold open : None On - location segment : Letterman visits Shangri La Studio , speaks
with music producer Rick Rubin and listens to Madison Ryann Ward recording . 5 `` TBA '' Tina Fey May 4 , 2018 6 `` TBA '' Howard Stern May 31 , 2018 Reception
( edit ) The first season of My Next Guest Needs No Introduction has been met with a positive response from critics . On the review aggregation website Rotten
Tomatoes , the first season holds a 80 % approval rating with an average rating of 6.4 out of 10 based on 25 reviews . Metacritic , which uses a weighted
average , assigned the season a score of 70 out of 100 based on 13 reviews , indicating `` generally favorable reviews '' . References ( edit ) Jump up ^
Andreeva , Nellie ( August 8 , 2017 ) . `` David Letterman Returns To Television With Netflix Series '' . Deadline .
</item 2>

<item 3>
My Next Guest Needs No Introduction with David Letterman
Rotten Tomatoes . Retrieved January 13 , 2018 . Jump up ^ `` My Next Guest Needs No Introduction With David Letterman : Season 1 - Metacritic '' . Metacritic .
Retrieved January 13 , 2018 . External links ( edit ) Official website David Letterman Television hosting The David Letterman Show ( 1980 ) Late Night with
David Letterman ( 1982 -- 1993 ) Late Show with David Letterman ( 1993 -- 2015 ) My Next Guest Needs No Introduction with David Letterman ( 2018 ) Companies The
Letterman Foundation for Courtesy and Grooming Rahal Letterman Lanigan Racing Worldwide Pants Related Dorothy Mengering Netflix original series Current Since
2013 House of Cards Orange Is the New Black Since 2014 BoJack Horseman Since 2015 Unbreakable Kimmy Schmidt Chef 's Table Daredevil Grace and Frankie Residue
Sense8 Club de Cuervos Project Mc Dinotrux Narcos Master of None Jessica Jones F Is for Family Making a Murderer Real Rob Since 2016 Degrassi : Next Class
Fuller House Lego Friends : The Power of Friendship Flaked
</item 3>

<item 4>
My Next Guest Needs No Introduction with David Letterman
The series consists of interviews with one guest per episode both inside and outside a studio setting . The first season will consist of six episodes and will
include interviews with Barack Obama , George Clooney , Malala Yousafzai , Jay - Z , Tina Fey , and Howard Stern . Obama was the first guest on the show , with
the first episode released on Netflix on January 12 , 2018 . Contents ( hide ) 1 Production 1.1 Development 1.2 Music 2 Episodes 3 Reception 4 References 5
External links Production ( edit ) Development ( edit ) On August 8 , 2017 , it was announced that Netflix had ordered a six - episode interview series starring
David Letterman . Each hour - long episode will feature Letterman conducting a longform conversation with a single guest and in - the - field segments , in
which he will explore topics on his own . On January 5 , 2018 , Netflix released further information about the series .
</item 4>

<item 5>
My Next Guest Needs No Introduction with David Letterman
2017 ) Glitch ( season 2 , 2017 ) Ended Lilyhammer ( 2012 -- 14 ) Hemlock Grove ( 2013 -- 15 ) Turbo FAST ( 2013 -- 16 ) VeggieTales in the House ( 2014 -- 16 )
Marco Polo ( 2014 -- 16 ) All Hail King Julien ( 2014 -- 17 ) The Adventures of Puss in Boots ( 2015 -- 18 ) Richie Rich ( 2015 ) Bloodline ( 2015 -- 17 )
Between ( 2015 -- 16 ) Wet Hot American Summer : First Day of Camp ( 2015 ) Popples ( 2015 -- 16 ) The Mr. Peabody & Sherman Show ( 2015 -- 17 ) Care Bears and
Cousins ( 2015 -- 16 ) W / Bob & David ( 2015 ) Atelier ( 2015 ) Dawn of the Croods ( 2015 -- 17 ) Chelsea Does ( 2016 ) Love ( 2016 -- 18 ) Lego Bionicle : The
Journey to One ( 2016 ) Netflix Presents :
</item 5>

<item 6>
My Next Guest Needs No Introduction with David Letterman
2017 ) Gypsy ( 2017 ) Wet Hot American Summer : Ten Years Later ( 2017 ) The Defenders ( 2017 ) Alias Grace ( 2017 ) Godless ( 2017 ) VeggieTales in the City (
2017 ) A StoryBots Christmas ( 2017 ) Damnation ( 2017 -- 18 ) Devilman Crybaby ( 2018 ) Collateral ( 2018 ) Everything Sucks ! ( 2018 ) Seven Seconds ( 2018 )
Upcoming A Little Help with Carol Burnett ( 2018 ) The Rain ( 2018 ) Safe ( 2018 ) The Break with Michelle Wolf ( 2018 ) Samantha ! ( 2018 ) The Innocents (
2018 ) 3 Below ( 2018 ) Baki ( 2018 ) Chilling Adventures of Sabrina ( 2018 ) Comedians in Cars Getting Coffee ( season 10 , 2018 ) Disenchantment ( 2018 )
Distrito Salvaje ( 2018 ) Dr.
</item 6>

<item 7>
My Next Guest Needs No Introduction with David Letterman
wikipedia.org/w/index.php?title=My_Next_Guest_Needs_No_Introduction_with_David_Letterman&oldid=838074810 '' Categories : 2010s American comedy television series
2018 American television series debuts David Letterman English - language television programs Netflix original programming Hidden categories : Pages using
infobox television with editor parameter Episode lists with unformatted air dates Talk Contents About Wikipedia Italiano Nederlands Português Edit links This
page was last edited on 24 April 2018 , at 19 : 28 . About Wikipedia
</item 7>

<item 8>
My Next Guest Needs No Introduction with David Letterman
Washington Post . Retrieved January 13 , 2018 . Jump up ^ Leight , Elias ( February 2 , 2018 ) . `` Paul Shaffer on Writing Letterman 's Netflix Theme Song &
the Value of Humor in Pop '' . Billboard . Retrieved February 9 , 2018 . Jump up ^ Carr , Mary Kate ( April 24 , 2018 ) . `` What 's on Netflix in May : The
best shows and movies to watch this month '' . Entertainment Weekly . Retrieved April 24 , 2018 . Jump up ^ Douglas , Nick ( April 24 , 2018 ) . `` Here 's What
's Coming And Going From Netflix In May 2018 '' . Lifehacker . Retrieved April 24 , 2018 . Jump up ^ `` My Next Guest Needs No Introduction With David Letterman
: Season 1 - Rotten Tomatoes '' . Rotten Tomatoes . Retrieved January 13 , 2018 .
</item 8>

<item 9>
My Next Guest Needs No Introduction with David Letterman
My Next Guest Needs No Introduction with David Letterman - wikipedia My Next Guest Needs No Introduction with David Letterman Jump to : navigation , search My
Next Guest Needs No Introduction with David Letterman Genre Talk show Directed by Michael Bonfiglio Presented by David Letterman Composer ( s ) Paul Shaffer
Country of origin United States Original language ( s ) English No. of seasons No. of episodes 4 ( list of episodes ) Production Executive producer ( s ) Justin
Wilkes Dave Sirulnick Jon Kamen Tom Keany Producer ( s ) Mary Barclay Cinematography Declan Quinn Editor ( s ) Jesse Gordon Edward A. Bishop Paul Little Running
time 50 -- 56 minutes Production company ( s ) RadicalMedia Worldwide Pants Inc . Release Original network Netflix Original release January 12 , 2018 ( 2018 -
01 - 12 ) -- present External links Official website My Next Guest Needs No Introduction with David Letterman is an American web television talk show hosted by
David Letterman . The series consists of interviews with one guest per episode both inside and outside a studio setting .
</item 9>

<item 10>
My Next Guest Needs No Introduction with David Letterman
Seuss ' Green Eggs and Ham ( 2018 ) Harvey Street Kids ( 2018 ) Hilda ( 2018 ) Insatiable ( 2018 ) Kingdom ( 2018 ) Kiss Me First ( 2018 ) Knights of the Zodiac
: Saint Seiya ( 2018 ) La Casa de las Flores ( 2018 ) Maniac ( 2018 ) The Ballad of Buster Scruggs ( 2018 ) The Epic Tales of Captain Underpants ( 2018 ) The
Umbrella Academy ( 2018 ) Carmen Sandiego ( 2019 ) Jinn ( 2019 ) Messiah ( 2019 ) Money Heist ( season 3 , 2019 ) Tales of the City ( 2019 ) Top Boy ( season 3
, 2019 ) The Last Kids on Earth ( 2019 ) Wizards ( 2019 ) Best Worst Weekend Ever ( TBA ) Chambers ( TBA ) Cursed ( TBA ) Dead to Me ( TBA ) Fast & Furious (
TBA ) Ghoul ( TBA ) Nightflyers ( TBA ) Norm Macdonald Has a Show ( TBA ) Paradise , P.
</item 10>
=== END ITEM id=e5b3029a32ae3df7 ===

=== ITEM id=398367a0435453ce ===
Gold sentences and facts:
Sentence 0: The Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the `` Watchtower '' and the `` Swamp '' .
  Fact 0.0: The Curse of Oak Island is a series.
  Fact 0.1: The Curse of Oak Island follows brothers Marty and Rick Lagina.
  Fact 0.2: Marty and Rick Lagina are originally from Kingsford, Michigan.
  Fact 0.3: The Curse of Oak Island follows the Laginas' efforts to find the speculated treasure or historical artifacts believed to be on Oak Island.
  Fact 0.4: The series discusses the history of Oak Island.
  Fact 0.5: The series discusses recent discoveries on Oak Island.
  Fact 0.6: The series discusses theories about Oak Island.
  Fact 0.7: The series discusses prior attempts to investigate the site on Oak Island.
  Fact 0.8: Areas of interest in the series include the "Money Pit".
  Fact 0.9: Areas of interest in the series include Borehole 10-x.
  Fact 0.10: Areas of interest in the series include Smith's Cove.
  Fact 0.11: Areas of interest in the series include "Nolan's Cross".
  Fact 0.12: Areas of interest in the series include the "Hatch".
  Fact 0.13: Areas of interest in the series include the "Watchtower".
  Fact 0.14: Areas of interest in the series include the "Swamp".

Retrieved items:
<item 1>
The Curse of Oak Island
Background ( edit ) The Lagina brothers became fascinated with the island after reading the January 1965 issue of Reader 's Digest magazine that features an
article on the Restall family 's work to investigate the so - called `` Money Pit '' . Marty and his brother Rick obtained a controlling interest in Oak Island
Tours , which reportedly owns most of the island . The brothers were later approached by Prometheus Entertainment to do a reality show . Rick and Marty have
engaged the assistance of father - and - son Dan and Dave Blankenship , permanent residents of the island who have likewise been searching for treasure since
the 1960s . Theories ( edit ) The series explores various Oak Island theories and conjectures through conversations with independent researchers .
</item 1>

<item 2>
The Curse of Oak Island
Series overview ( edit ) Season Episodes Originally aired First aired Last aired 5 January 5 , 2014 ( 2014 - 01 - 05 ) February 9 , 2014 ( 2014 - 02 - 09 ) 10
November 4 , 2014 ( 2014 - 11 - 04 ) January 13 , 2015 ( 2015 - 01 - 13 ) 13 November 10 , 2015 ( 2015 - 11 - 10 ) February 2 , 2016 ( 2016 - 02 - 02 ) 16
November 15 , 2016 ( 2016 - 11 - 15 ) February 21 , 2017 ( 2017 - 02 - 21 ) 5 18 November 7 , 2017 ( 2017 - 11 - 07 ) March 6 , 2018 ( 2018 - 03 - 06 )
Production ( edit ) Having started in 2014 , The Curse of Oak Island is currently in its fifth season . On August 2 , 2016 , it was announced that Nova Scotia
Business Inc.
</item 2>

<item 3>
*The Curse of Oak Island* follows Michigan brothers Rick and Marty Lagina, with longtime searchers Dan and Dave Blankenship, investigating Oak Island’s rumored
treasure and history. Since its 2014 premiere, the team has explored the Money Pit, Smith’s Cove, Nolan’s Cross, the swamp, and boreholes 10-X and C-1. Finds
include coins, pottery, wood, metal, bones, and a small lead cross; tests dated some wood and bone to the
</item 3>

<item 4>
The Curse of Oak Island
overall No. in season Title Original air date Viewers ( millions ) 29 `` Going for Broke '' November 15 , 2016 ( 2016 - 11 - 15 ) 2.62 The team agrees to turn
focus over to the Money Pit and review what they have collected . Researcher Zena Halpern presents a handmade copy of what she claims is a French map from 1347
of Oak Island . The map shows locations for a `` hatch '' , a `` valve '' and an `` anchor '' on the Island . A theory is presented based on a purported
decoding of a cipher that Joab at the time of King David transported the Ark of the Covenant to North Africa from which it was later transported onward by the
Knights Templar from the then - Portuguese town of Safi in Morocco to Oak Island . 30 `` Always Forward '' November 22 , 2016 ( 2016 - 11 - 22 ) 2.66 While
preparing to excavate the Money Pit , the team investigates an 18th century well on mainland Nova Scotia .
</item 4>

<item 5>
The Curse of Oak Island
On August 2 , 2016 , it was announced that Nova Scotia Business Inc. approved $1,271,546 in film funding for the production of the 4th season . In October 2017
, a fifth season was announced , which began airing on November 7 , 2017 . Episodes ( edit ) Season 1 ( 2014 ) ( edit ) No . overall No. in season Title
Original air date Viewers ( millions ) `` What Lies Below '' January 5 , 2014 ( 2014 - 01 - 05 ) 2.53 Brothers Rick and Marty Lagina now own most of Oak Island
. Determined to resume the search for treasure on the island , they explore an abandoned shaft called Borehole 10 - X and use water pressure to pump material to
the surface . They find a tiny piece of metal .
</item 5>

<item 6>
*The Curse of Oak Island* is a reality series about efforts to solve Oak Island’s mystery by searching for historical artifacts and treasure. Narrated by Robert
Clotworthy, it premiered on History Canada on January 5, 2014. Brothers Rick and Marty Lagina, who own most of the island, lead the search. In the first
episode, they investigate Borehole 10-X and recover a small piece of metal. The series’ fifth season aired from November 2017 to
</item 6>

<item 7>
The Curse of Oak Island
61 17 `` A Family Album '' February 27 , 2018 ( 2018 - 02 - 27 ) 1.98 The episode includes accounts of activities on the Oak Island by the McGinnis family ,
descendants of Samuel Ball , William Chappell 's family , the Restall Family and the family of Robert Dunfield . The Blankenship / Nolan disputes on Oak Island
are discussed . The episode ends with a short history of the Lagina brothers ' interest in Oak Island . Note : This episode is a special . 62 18 `` Amazing
Discoveries '' March 6 , 2018 ( 2018 - 03 - 06 ) 2.89 The gem in the Jewelry fragment found on the island is a Rhodolite Garnet stone of small value . The
drilling obstruction in the `` money pit '' borehole is determined by divers to be a granite boulder . References ( edit ) ^ Jump up to : `` Shows A-Z - curse
of oak island , the on history '' . The Futon Critic . Retrieved December 6 , 2017 .
</item 7>

<item 8>
*The Curse of Oak Island* follows Michigan brothers Rick and Marty Lagina as they investigate rumored treasure and artifacts on Nova Scotia’s Oak Island.
Inspired by a 1965 *Reader’s Digest* article, they acquired a controlling interest in Oak Island Tours and enlisted longtime searchers Dan and Dave Blankenship.
The series covers the island’s history, past expeditions, and theories, focusing on sites such as the Money Pit, Smith’s Cove, Nolan’s Cross, and the Swamp.
Spec
</item 8>

<item 9>
The Curse of Oak Island
Contents ( hide ) 1 Overview 1.1 Background 1.2 Theories 2 Series overview 3 Production 4 Episodes 4.1 Season 1 ( 2014 ) 4.2 Season 2 ( 2014 -- 2015 ) 4.3
Season 3 ( 2015 -- 2016 ) 4.4 Season 4 ( 2016 -- 2017 ) 4.5 Season 5 ( 2017 -- 2018 ) 5 References 6 Further reading 7 External links Overview ( edit ) The
Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or
historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to
investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the ``
Watchtower '' and the `` Swamp '' .
</item 9>

<item 10>
The Curse of Oak Island
The Curse of Oak Island - wikipedia The Curse of Oak Island Jump to : navigation , search The Curse of Oak Island Genre Reality Narrated by Robert Clotworthy
Country of origin Canada Original language ( s ) English No. of seasons 5 No. of episodes 62 Production Executive producer ( s ) Kevin Burns Camera setup
Multiple Running time 43 minutes Production company ( s ) Prometheus Entertainment Distributor A&E Networks Release Original network History Canada Original
release January 5 , 2014 ( 2014 - 01 - 05 ) -- present External links Website Production website The Curse of Oak Island is an active reality television series
that first premiered in Canada on the History network on January 5 , 2014 . The show features what is known as the Oak Island mystery , showing efforts to
search for historical artifacts and treasure .
</item 10>
=== END ITEM id=398367a0435453ce ===

=== ITEM id=5055652b5c3d4dea ===
Gold sentences and facts:
Sentence 0: Gina Tognoni / toʊnˈjoʊni / ( born November 28 , 1973 ) is an American actress , best known for her work with American daytime soap operas . Her most notable performances include Kelly Cramer on One Life to Live and Dinah Marler on Guiding Light . She is currently starring as Phyllis Summers on The Young and the Restless .
  Fact 0.0: Gina Tognoni was born on November 28, 1973.
  Fact 0.1: Gina Tognoni is an American actress.
  Fact 0.2: Gina Tognoni is best known for her work with American daytime soap operas.
  Fact 0.3: Gina Tognoni's most notable performances include Kelly Cramer on One Life to Live.
  Fact 0.4: Gina Tognoni's most notable performances include Dinah Marler on Guiding Light.
  Fact 0.5: Gina Tognoni is currently starring as Phyllis Summers on The Young and the Restless.
Sentence 1: Gina Tognoni ( 1973 - 11 - 28 ) November 28 , 1973 ( age 44 ) St. Louis , Missouri , U.S. Occupation Actress businesswoman Years active 1995 -- present Television Guiding Light ( 2004 -- 2009 ) One Life to Live ( 1995 -- 2001 , 2002 , 2010 -- 2011 ) The Young and the Restless ( 2014 -- present ) Spouse ( s ) Joseph Chiarello ( m . 2009 )
  Fact 1.0: Gina Tognoni was born on November 28, 1973.
  Fact 1.1: Gina Tognoni is age 44.
  Fact 1.2: Gina Tognoni was born in St. Louis, Missouri, U.S.
  Fact 1.3: Gina Tognoni's occupations are actress and businesswoman.
  Fact 1.4: Gina Tognoni's years active are 1995 to present.
  Fact 1.5: Gina Tognoni was on Guiding Light from 2004 to 2009.
  Fact 1.6: Gina Tognoni was on One Life to Live from 1995 to 2001, in 2002, and from 2010 to 2011.
  Fact 1.7: Gina Tognoni was on The Young and the Restless from 2014 to present.
  Fact 1.8: Gina Tognoni's spouse is Joseph Chiarello.
  Fact 1.9: Gina Tognoni married Joseph Chiarello in 2009.

Retrieved items:
<item 1>
Sheila Carter
After Lauren arrived she was tied up , and Sheila flees with Phyllis and babies Fen and Summer as hostages . By this time , Michael arrives at the warehouse and
frees Paul and Maggie , the latter being taken to the hospital . Posing as an elderly woman named Betsy , Sheila takes Phyllis , Summer , and Fen to a nearby
retirement home to hide out . After learning of their location , Lauren goes to that home carrying a gun in her purse . Following a tip from a neighbor , Lauren
finds Sheila 's apartment and forces her way inside . Sheila and Phyllis then argue over who is who , and Lauren screams at them to stop . Sheila then runs to
grab baby Summer in order to use her as a shield against Lauren ; Phyllis tells Lauren to point the gun at her to protect her daughter , and a frightened Lauren
, realizing the real Phyllis would never put her daughter in the line of fire , turns to Sheila and shoots her . Sheila is then declared dead at the scene .
2010 , 2012 ( edit ) In 2010 , Sarah Smythe , Sheila Carter 's sister , arrives in town .
</item 1>

<item 2>
Sheila Carter
She has also had conflicts with Stephanie Forrester , Maggie Forrester , Amber Moore , Taylor Hayes , Brooke Logan , Phyllis Summers and Quinn Fuller . Contents
( hide ) 1 Casting and creation 2 Development 3 Storylines 3.1 1990 -- 92 3.2 1992 -- 98 3.3 2002 -- 03 3.4 2005 -- 07 3.5 2010 , 2012 3.6 2017 -- 4 Critical
reception 5 References 6 External links Casting and creation ( edit ) The role was portrayed by Kimberlin Brown for a total of 16 years . Her first run was on
The Young and the Restless from May 1990 to May 20 , 1992 , when the character crossed over to The Bold and the Beautiful from May 21 , 1992 , to October 2 ,
1998 , again from May 24 to November 5 , 2002 , and again finally from September 26 to October 10 , 2003 .
</item 2>

<item 3>
Sheila Carter
( 1991 -- 92 ) Eric Forrester ( 1993 -- 95 ) James Warwick ( 1997 -- 98 ) Sons Ryder Callahan Daughters Mary Carter Warwick Diana Carter Daisy Carter
Granddaughters Lucy Romalotti Sheila Carter is a fictional character from The Young and the Restless and The Bold and the Beautiful , American soap operas on
the CBS network . Created by William J. Bell , the role was introduced in 1990 -- by Edward J. Scott -- under the portrayal of Kimberlin Brown , who portrayed
the role for many intervals on both soaps until 2006 . That same year , Michelle Stafford took over the role , after Sheila had plastic surgery to look like
Phyllis Summers . Brown returned to the role of Sheila on The Bold and the Beautiful , beginning on June 9 , 2017 . Sheila is known as a villain . A significant
portion of Sheila 's history on both soaps revolves around her long - running conflict with Lauren Fenmore , whom she has attempted to kill on multiple
occasions .
</item 3>

<item 4>
Sheila Carter
Sheila continues to insinuate herself into Eric 's life , faking a concussion after a catfight with Quinn to manipulate Eric into letting her stay in the
Forrester mansion . Sheila blackmails her ex-husband , James Warwick , into lying about the severity of her condition . Critical reception ( edit ) Sheila is
often considered to be one of the most popular villains in daytime . Kimberlin Brown was nominated for a Daytime Emmy in the role as Outstanding Supporting
Actress in 1993 . With the global success of The Bold and the Beautiful , Sheila is arguably the most recognisable soap opera villain in the world . References
( edit ) ^ Jump up to : `` About Y&R : Who 's Who in Los Angeles - Sheila Carter '' . SoapCental . Retrieved 16 September 2012 . ^ Jump up to : Kroll , Dan J. (
9 July 2005 ) . `` Y&R confirms the return of Kimberlin Brown '' . SoapCentral . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday
, 5th August 2005 '' . CBS.com .
</item 4>

<item 5>
Gina Tognoni
She is currently starring as Phyllis Summers on The Young and the Restless . Contents ( hide ) 1 Career 2 Personal life 3 Filmography 4 Awards and nominations 5
References 6 External links Career ( edit ) Tognoni is known for her roles on daytime television soap operas . In 1995 , she made her soap debut as Kelly Cramer
in ABC 's One Life to Live . She appeared on the show until 2001 . In April 2001 , it was announced that Tognoni , after six years with the show , had decided
not to renew her contract with the series . Tognoni returned to daytime television as Dinah Marler on CBS 's Guiding Light in July 2004 , and won her first
Daytime Emmy Award for Outstanding Supporting Actress in a Drama Series in 2006 for her work on the show . In 2007 , she was nominated again in the same
category but lost to Genie Francis . She won her second Emmy in 2008 , becoming only the second actress to be a repeat winner in the Supporting Actress category
.
</item 5>

<item 6>
Gina Tognoni
( 2009 - 06 - 08 ) . `` Gina Tognoni close to deciding on her post-GL career '' . SoapCentral . Retrieved June 28 , 2012 . Jump up ^ `` AMC and OLTL Canceled ''
. Soap Opera Digest . Soapoperadigest.com. 2011 - 04 - 14 . Retrieved June 28 , 2012 . Jump up ^ Kroll , Dan J. ( 2011 - 06 - 07 ) . `` Gina Tognoni , Tom
Degnan exiting OLTL this summer '' . SoapCentral . Retrieved June 28 , 2012 . Jump up ^ Fairman , Michael ( 2011 - 05 - 31 ) . `` Is Gina Tognoni headed out of
One Life to Live early ? '' . Michael Fairman On - Air On - Soaps . Retrieved June 28 , 2012 . Jump up ^ SOD ( May 22 , 2014 ) . `` GINA TOGNONI IS Y&R 'S NEW
PHYLLIS ! '' . Soap Opera Digest .
</item 6>

<item 7>
Sheila Carter
However , that December , the character returned , though not portrayed by Brown , but by Michelle Stafford , who in addition portrayed Phyllis Summers . For
the reason of Brown not returning to the role , the actress said : `` I had a bad accident last year , and when they wanted me back in November , I really could
n't go at that time , and that 's why I came back looking like somebody else . '' Stafford only portrayed the role for two months until February 9 , 2007 , when
Sheila was shot by her longtime enemy Lauren Fenmore and presumed dead . In 2009 , reports speculated that Brown would reprise the role again on The Bold and
the Beautiful , which was followed by more rumors speculating a return to The Young and the Restless . `` I actually got a call from ( B&B head writer and
executive producer ) Brad Bell a while back asking if I 'd be interested in coming back , and I said , ' Sure ! ' The thing with daytime is that the storylines
are ( written ) so far in advance , you just never know when ( you might be needed ) , '' Brown said at the time .
</item 7>

<item 8>
Sheila Carter
Having herself had plastic surgery to resemble Lauren , Sarah holds both Phyllis and Lauren responsible for her sister 's death . Sarah devises a plan with
Daisy Carter and Ryder Callahan , Sheila 's twins with Tom Fisher , to take over Lauren 's life and riches . The plan fails , with Lauren shooting and killing
Sarah in self - defense to Sarah 's attempt to fire her gun at Phyllis . In 2012 , a woman claiming to be Daisy 's mother checks her out of a mental institution
, signing the name `` Sheila Carter '' on the release form . Michael has said signature analyzed to see if it fits Sheila 's handwriting , and results turn out
to be inconclusive . 2017 -- ( edit ) Sheila ( Brown ) returns to Los Angeles in June 2017 . When she attempts to leave Katie Logan ( Heather Tom ) 's home
without being noticed , Quinn Forrester ( Rena Sofer ) hears her leaving , immediately asking who she is . Introducing herself as a friend of the Logan family ,
Sheila reveals herself to Katie , making for her assistance in making amends with those she had wronged in Los Angeles .
</item 8>

<item 9>
Sheila Carter
Sheila Carter - Wikipedia Sheila Carter Jump to : navigation , search Sheila Carter Kimberlin Brown as Sheila Carter The Young and the Restless character
Portrayed by Kimberlin Brown ( 1990 -- ) Michelle Stafford ( 2006 -- 07 ) Duration 1990 -- 98 2002 -- 03 2005 -- 07 2017 -- First appearance May 1990 Created by
William J. Bell Introduced by Edward J. Scott ( 1990 , 1992 -- 95 ) Lee Phillip Bell ( 1992 ) Bradley Bell ( 2002 -- 03 ) John F. Smith ( 2005 ) Lynn Marie
Latham and Josh Griffith ( 2006 ) Bradley Bell and Michael Minnis ( 2017 ) Crossover appearances The Bold and the Beautiful Classification Present ; regular
Profile Other names Sheila Grainger Sheila Forrester Sheila Warwick Sybil Weller Sheila Lovejoy Brenda Harris Jennifer Mitchell Phyllis Newman Betsy Occupation
Waitress Executive assistant Nurse Residence Los Angeles , California Michelle Stafford as Sheila Carter ( show ) Family Family Carter Mother Molly Carter
Sisters Sarah Smythe Husband Scott Grainger , Sr.
</item 9>

<item 10>
Sheila Carter
and Colleen Daniel and Lily Devon and Hilary `` Nadia 's Theme '' ( hide ) The Young and the Restless characters Present and future characters Ashley Abbott
Billy Abbott Jack Abbott Jill Abbott John Abbott Johnny Abbott Traci Abbott Cane Ashby Charlie Ashby Mattie Ashby Sam Ashby Lily Winters Colin Atkinson Fenmore
Baldwin Michael Baldwin Gloria Abbott Bardwell Jeffrey Bardwell Christine Blair Mariah Copeland Hilary Curtis Lauren Fenmore Kevin Fisher Scotty Grainger Devon
Hamilton Reed Hellstrom Anita Lawson Dina Mergeron Chloe Mitchell Leslie Michaelson Abby Newman Chelsea Lawson Christian Newman Connor Newman Faith Newman Katie
Newman Nicholas Newman Nikki Newman Noah Newman Sharon Newman Summer Newman Victor Newman Victoria Newman Stitch Rayburn Barton Shelby Phyllis Summers Esther
Valentine Paul Williams Neil Winters ( hide ) Past characters Kyle Abbott Kelly Andrews Genevieve Atkinson Avery Bailey Clark Eden Baldwin Isabella Braña
Mackenzie Browning Brad Carlton Colleen Carlton Daisy Carter Sheila Carter Chance Chancellor Katherine Chancellor Joe Clark Matt Clark Greg Foster Dr. Snapper
Foster Sam Gibson Diego Guittierez Nathan Hastings Jana Hawkes J.T.
</item 10>
=== END ITEM id=5055652b5c3d4dea ===

=== ITEM id=d5e217134d2aba56 ===
Gold sentences and facts:
Sentence 0: The song relates to both the Old and New Testaments . The verses reflect the Israelites ' escape out of Egypt as found in Exodus 14 . The chorus refers to healing : see John 5 : 4 , `` For an angel went down at a certain season into the pool , and troubled the water : whosoever then first after the troubling of the water stepped in was made whole of whatsoever disease he had . ''
  Fact 0.0: The song relates to both the Old Testament and the New Testament.
  Fact 0.1: The verses of the song reflect the Israelites' escape out of Egypt as found in Exodus 14.
  Fact 0.2: The chorus of the song refers to healing.
  Fact 0.3: The chorus refers to John 5:4.
  Fact 0.4: John 5:4 says an angel went down at a certain season into the pool and troubled the water.
  Fact 0.5: John 5:4 says whoever first stepped into the pool after the water was troubled was made whole of whatever disease he had.

Retrieved items:
<item 1>
Wade in the Water
Retrieved 2016 - 09 - 29 . Jump up ^ `` Wade in the Water , Children ( DVD ) Elizabeth Wood , Gabriel Nussbaum '' . IndiePix Films . Retrieved 2016 - 09 - 29 .
Jump up ^ William Edward Burghardt Du Bois ( 1925 ) . Crisis . Crisis Publishing Company . Jump up ^ Robert M. Marovich ( 15 March 2015 ) . A City Called Heaven
: Chicago and the Birth of Gospel Music . University of Illinois Press . p. 135 . ISBN 978 - 0 - 252 - 09708 - 9 . Sources ( edit ) Waltz , Robert B ; David G.
Engle . `` Wade In The Water '' . The Traditional Ballad Index : An Annotated Bibliography of the Folk Songs of the English - Speaking World . Hosted by
California State University , Fresno , Folklore , 2007 . Further reading Jones , Arthur C. ( 2005 ) . Wade in the water : the wisdom of the spirituals ( 3rd ed
. ) .
</item 1>

<item 2>
Wade in the Water
Wade in the Water - wikipedia Wade in the Water Jump to : navigation , search This article is about the song . For the album by Ramsey Lewis , see Wade in the
Water ( album ) . This article needs additional citations for verification . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed . ( September 2016 ) ( Learn how and when to remove this template message ) `` Wade in the Water '' American
folk song Native language ( s ) English First publication New Jubilee Songs as Sung by the Fisk Jubilee Singers ( 1901 ) Catalogue ( s ) Roud 5439 Notable
performer ( s ) The Staple Singers , Ramsey Lewis ; etc . First released recording Title of the recording `` Good News Chariot 's Coming and Wade in the Water
'' Released ca .
</item 2>

<item 3>
Wade in the Water
. Boulder , Colo. : Leave a Little Room . pp. 54 , 112 . ISBN 0976237709 . `` Music : Wade in the Water '' . Pathways to Freedom : Maryland & the Underground
Railroad Secrets : Signs and Symbols . Maryland Public Television . 2014 . Tobin , Jacqueline L. ; Dobard , Raymond G. ( 2011 ) . Hidden in plain view the
secret story of quilts and the underground railroad . New York , N.Y. : Anchor Books . p. 126 . ISBN 0307790568 .
</item 3>

<item 4>
Wade in the Water
Many Internet sources and popular books claim that songs such as `` Wade in the Water '' contained explicit instructions to fugitive slaves on how to avoid
capture and the route to take to successfully make their way to freedom . An example of this is cited in the book Pathways to Freedom : Maryland & the
Underground Railroad . The book explains how Harriet Tubman used the song `` Wade in the Water '' to tell escaping slaves to get off the trail and into the
water to make sure that the dogs employed by slavers lost their trail . In popular culture ( edit ) One version of the song performed by The Staple Singers
became a part of the civil rights movement in the US . `` Wade in the Water '' was a popular instrumental hit in 1966 for the Ramsey Lewis Trio , which prompted
further instrumental recordings by Herb Alpert and the Tijuana Brass , and Billy Preston ( both 1967 ) . The melody was used for the 1988 Tony ! Toni ! Toné !
hit `` Little Walter '' .
</item 4>

<item 5>
Wade in the Water
The story follows the lives of three characters before , during , and after the great Johnstown Flood of 1889 . Selective list of recordings ( edit ) This
section does not cite any sources . Please help improve this section by adding citations to reliable sources . Unsourced material may be challenged and removed
. ( February 2017 ) ( Learn how and when to remove this template message ) The first commercially recorded version of the song was released by Paramount Records
( sung by Sunset Four Jubilee Singers in 1925 ) , as `` Good News Chariot 's Coming and Wade in the Water '' .
</item 5>

<item 6>
Wade in the Water
2004 ) includes the chorus of `` Wade in the Water '' with the altered lyrics `` Straight from the Water , Straight from the water , children '' IIIrd Tyme Out
( The Best Durn Ride ) ( 2004 ) Mavis Staples ( Live : Hope at the Hideout ) ( 2008 ) J. Boykin ( Jesus Loves Me ) ( 2009 ) John Boutte , recorded for the film
Wade in the Water , Children ( see above ) Patty Griffin Downtown Church ( 2010 ) Cerys Matthews - `` Explorer '' ( 2011 ) Tedeschi Trucks Band Everybody 's
Talkin ' ( 2012 ) Aleksander Vinter sampled the first verse in a song of the same title ( 2012 ) Tina Savage The Son and the Soul ( 2013 ) Jamie N Commons
Spotify Sessions ( Jamie N Commons EP ) ( 2013 ) Frances Madden ( 2014 ) Sirena : the Sirens Out of the Blue ( 2014 ) PJ Harvey as an intro to `` River
Anacostia '' ( 2016 ) References ( edit ) ^ Jump up to : Steve Sullivan ( 17 May 2017 ) .
</item 6>

<item 7>
Wade in the Water
: Anchor Books . p. 126 . ISBN 0307790568 . External links ( edit ) Lyrics at Negrospirituals.org MIDI at Gospelsonglyrics.org Lyrics of this song at
MetroLyrics Retrieved from `` https://en.wikipedia.org/w/index.php?title=Wade_in_the_Water&oldid=833167717 '' Categories : American folk songs Bob Dylan songs
African - American spiritual songs Eva Cassidy songs 1966 singles 1967 singles Year of song unknown Songs based on American history Hidden categories : Articles
needing additional references from September 2016 All articles needing additional references All articles with specifically marked weasel - worded phrases
Articles with specifically marked weasel - worded phrases from November 2015 Articles needing additional references from February 2017 Talk Contents About
Wikipedia Deutsch Italiano Edit links This page was last edited on 30 March 2018 , at 01 : 17 . Text is available under the Creative Commons Attribution -
ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark
of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Wade in the Water
1925 Record label Paramount Records Artist Sunset Four Jubilee Singers Genre Negro spiritual `` Wade in the Water '' ( Roud 5439 ) is the name of a Negro
spiritual first published in New Jubilee Songs as Sung by the Fisk Jubilee Singers ( 1901 ) by John Wesley Work II and his brother , Frederick J. Work ( see
Fisk Jubilee Singers ) . It is associated with the songs of the Underground Railroad . Wade in the water Wade in the water , children Wade in the water God 's
gon na trouble the water Contents ( hide ) 1 Meaning 2 In popular culture 3 Selective list of recordings 4 References 4.1 Sources 5 External links Meaning (
edit ) The song relates to both the Old and New Testaments . The verses reflect the Israelites ' escape out of Egypt as found in Exodus 14 . The chorus refers
to healing : see John 5 : 4 , `` For an angel went down at a certain season into the pool , and troubled the water : whosoever then first after the troubling of
the water stepped in was made whole of whatsoever disease he had . ''
</item 8>

<item 9>
Wade in the Water
Peace and Happiness ( 1969 ) CCS ( 1970 ) Alex Harvey , with Rock Workshop ( 1971 ) Peter Herbolzheimer with Rhythm Combination & Brass ( 1972 ) Ellen McIlwain
Honkey Tonk Angel ( Polydor , 1972 ) Fine Young Cannibals , B - side to Blue ( 1985 ) Tony ! Toni ! Toné ! 's song `` Little Walter '' ( Who ? , 1988 ) includes
the chorus of `` Wade in the Water '' with the altered lyrics `` Hey , Little Walter ... Hey , Little Walter , listen ... Hey , Little Walter ... Somethin 's
gon na getcha , Little Walter '' Eva Cassidy ( 1997 ) Make - up Wade in the Water 7 '' ( 1997 ) Chanticleer , recorded in `` Reflections ( An Anniversary
Celebration ) '' ( 1997 ) and `` A Portrait '' ( 2005 ) Mary Mary Thankful ( 2000 ) Moses Hogan Moses Hogan Singers : Choral Series ( 2002 ) Marc Broussard 's
song `` Home '' ( Carenco ,
</item 9>

<item 10>
Wade in the Water
Encyclopedia of Great Popular Song Recordings . Rowman & Littlefield Publishers . p. 387 . ISBN 978 - 1 - 4422 - 5449 - 7 . Jump up ^ `` Wade in De Water '' .
Archived from the original on July 9 , 2008 . Retrieved January 11 , 2008 . Jump up ^ `` Archived copy '' . Archived from the original on 2008 - 07 - 24 .
Retrieved 2008 - 08 - 18 . Jump up ^ `` Pathways to Freedom Secrets : Signs and Symbols Music '' . Pathways.thinkport.org . Retrieved 2016 - 09 - 29 . Jump up ^
Robert Pruter ( 1992 ) . Chicago Soul . University of Illinois Press . p. 125 . ISBN 978 - 0 - 252 - 06259 - 9 . Jump up ^ `` Wade in the Water : A Must See
Film : Boston Globe '' . Wadeinthewater.com . Retrieved 2016 - 09 - 29 .
</item 10>
=== END ITEM id=d5e217134d2aba56 ===
