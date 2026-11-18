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

=== ITEM id=75334eadf2d9bbd0 ===
Gold sentences and facts:
Sentence 0: Brain tissue consumes a large amount of energy in proportion to its volume , so large brains place severe metabolic demands on animals . The need to limit body weight in order , for example , to fly , has apparently led to selection for a reduction of brain size in some species , such as bats . Most of the brain 's energy consumption goes into sustaining the electric charge ( membrane potential ) of neurons . Most vertebrate species devote between 2 % and 8 % of basal metabolism to the brain . In primates , however , the percentage is much higher -- in humans it rises to 20 -- 25 % . The energy consumption of the brain does not vary greatly over time , but active regions of the cerebral cortex consume somewhat more energy than inactive regions ; this forms the basis for the functional brain imaging methods PET , fMRI , and NIRS . The brain typically gets most of its energy from oxygen - dependent metabolism of glucose ( i.e. , blood sugar ) , but ketones provide a major alternative source , together with contributions from medium chain fatty acids ( caprylic and heptanoic acids ) , lactate , acetate , and possibly amino acids .
  Fact 0.0: Brain tissue consumes a large amount of energy in proportion to its volume.
  Fact 0.1: Large brains place severe metabolic demands on animals.
  Fact 0.2: The need to limit body weight, for example in order to fly, has apparently led to selection for a reduction of brain size in some species.
  Fact 0.3: Bats are one of the species that have apparently been selected for reduced brain size.
  Fact 0.4: Most of the brain's energy consumption goes into sustaining the electric charge (membrane potential) of neurons.
  Fact 0.5: Most vertebrate species devote between 2% and 8% of basal metabolism to the brain.
  Fact 0.6: In primates, the percentage of basal metabolism devoted to the brain is much higher.
  Fact 0.7: In humans, the brain's share of basal metabolism rises to 20-25%.
  Fact 0.8: The energy consumption of the brain does not vary greatly over time.
  Fact 0.9: Active regions of the cerebral cortex consume somewhat more energy than inactive regions.
  Fact 0.10: This difference in energy use forms the basis for the functional brain imaging methods PET, fMRI, and NIRS.
  Fact 0.11: The brain typically gets most of its energy from oxygen-dependent metabolism of glucose (blood sugar).
  Fact 0.12: Ketones provide a major alternative energy source for the brain.
  Fact 0.13: Medium chain fatty acids (caprylic and heptanoic acids), lactate, acetate, and possibly amino acids also contribute to the brain's energy supply.

Retrieved items:
<item 1>
Brain
Most of the brain 's energy consumption goes into sustaining the electric charge ( membrane potential ) of neurons . Most vertebrate species devote between 2 %
and 8 % of basal metabolism to the brain . In primates , however , the percentage is much higher -- in humans it rises to 20 -- 25 % . The energy consumption of
the brain does not vary greatly over time , but active regions of the cerebral cortex consume somewhat more energy than inactive regions ; this forms the basis
for the functional brain imaging methods PET , fMRI , and NIRS . The brain typically gets most of its energy from oxygen - dependent metabolism of glucose (
i.e. , blood sugar ) , but ketones provide a major alternative source , together with contributions from medium chain fatty acids ( caprylic and heptanoic acids
) , lactate , acetate , and possibly amino acids . Functions Model of a neural circuit in the cerebellum , as proposed by James S. Albus . Information from the
sense organs is collected in the brain . There it is used to determine what actions the organism is to take . The brain processes the raw data to extract
information about the structure of the environment .
</item 1>

<item 2>
Brain
Jump up ^ Armstrong , E ( 1983 ) . `` Relative brain size and metabolism in mammals '' . Science . 220 ( 4603 ) : 1302 -- 1304 . Bibcode : 1983Sci ... 220.1302
A . doi : 10.1126 / science. 6407108 . PMID 6407108 . Jump up ^ Jerison , HJ ( 1973 ) . Evolution of the Brain and Intelligence . Academic Press . pp. 55 -- 74
. ISBN 978 - 0 - 12 - 385250 - 2 . Jump up ^ Parent , A ; Carpenter , MB ( 1995 ) . `` Ch. 1 '' . Carpenter 's Human Neuroanatomy . Williams & Wilkins . ISBN
978 - 0 - 683 - 06752 - 1 . Jump up ^ Pardridge , W ( 2005 ) . `` The Blood - Brain Barrier : Bottleneck in Brain Drug Development '' . NeuroRx. 2 ( 1 ) : 3 --
14 .
</item 2>

<item 3>
Brain
PMID 7282965 . Jump up ^ Raichle , M ; Gusnard , DA ( 2002 ) . `` Appraising the brain 's energy budget '' . Proc. Natl. Acad. Sci. U.S.A. 99 ( 16 ) : 10237 --
10239 . Bibcode : 2002PNAS ... 9910237R . doi : 10.1073 / pnas. 172399499 . PMC 124895 . PMID 12149485 . Jump up ^ Mehagnoul - Schipper , DJ ; Van Der Kallen ,
BF ; Colier , WNJM ; Van Der Sluijs , MC ; Van Erning , LJ ; Thijssen , HO ; Oeseburg , B ; Hoefnagels , WH ; Jansen , RW ( 2002 ) . `` Simultaneous
measurements of cerebral oxygenation changes during brain activation by near - infrared spectroscopy and functional magnetic resonance imaging in healthy young
and elderly subjects '' . Hum Brain Mapp. 16 ( 1 ) : 14 -- 23 .
</item 3>

<item 4>
Brain
In other respects , the quantity and quality of experience are important ; for example , there is substantial evidence that animals raised in enriched
environments have thicker cerebral cortices , indicating a higher density of synaptic connections , than animals whose levels of stimulation are restricted .
Physiology. The functions of the brain depend on the ability of neurons to transmit electrochemical signals to other cells , and their ability to respond
appropriately to electrochemical signals received from other cells . The electrical properties of neurons are controlled by a wide variety of biochemical and
metabolic processes , most notably the interactions between neurotransmitters and receptors that take place at synapses . Neurotransmitters and receptors
Neurotransmitters are chemicals that are released at synapses when an action potential activates them -- neurotransmitters attach themselves to receptor
molecules on the membrane of the synapse 's target cell , and thereby alter the electrical or chemical properties of the receptor molecules . With few
exceptions , each neuron in the brain releases the same chemical neurotransmitter , or combination of neurotransmitters , at all the synaptic connections it
makes with other neurons ; this rule is known as Dale 's principle . Thus , a neuron can be characterized by the neurotransmitters that it releases .
</item 4>

<item 5>
Brain
Norepinephrine , which is involved in arousal , comes exclusively from a nearby small area called the locus coeruleus . Other neurotransmitters such as
acetylcholine and dopamine have multiple sources in the brain , but are not as ubiquitously distributed as glutamate and GABA . Electrical activity Brain
electrical activity recorded from a human patient during an epileptic seizure . As a side effect of the electrochemical processes used by neurons for signaling
, brain tissue generates electric fields when it is active . When large numbers of neurons show synchronized activity , the electric fields that they generate
can be large enough to detect outside the skull , using electroencephalography ( EEG ) or magnetoencephalography ( MEG ) . EEG recordings , along with
recordings made from electrodes implanted inside the brains of animals such as rats , show that the brain of a living animal is constantly active , even during
sleep . Each part of the brain shows a mixture of rhythmic and nonrhythmic activity , which may vary according to behavioral state .
</item 5>

<item 6>
Brain
2012.151 . PMC 3564188 . PMID 23072752 . Jump up ^ Boumezbeur , F. ; Petersen , KF. ; Cline , GW. ; Mason , GF. ; Behar , KL. ; Shulman , GI. ; Rothman , DL. (
Oct 2010 ) . `` The contribution of blood lactate to brain energy metabolism in humans measured by dynamic 13C nuclear magnetic resonance spectroscopy '' . J
Neurosci. 30 ( 42 ) : 13983 -- 91 . doi : 10.1523 / JNEUROSCI. 2040 - 10.2010 . PMC 2996729 . PMID 20962220 . Jump up ^ Deelchand , DK. ; Shestov , AA. ; Koski
, DM. ; Uğurbil , K. ; Henry , PG . ( May 2009 ) . `` Acetate transport and utilization in the rat brain '' . J Neurochem.
</item 6>

<item 7>
Brain
In mammals , the cerebral cortex tends to show large slow delta waves during sleep , faster alpha waves when the animal is awake but inattentive , and chaotic -
looking irregular activity when the animal is actively engaged in a task . During an epileptic seizure , the brain 's inhibitory control mechanisms fail to
function and electrical activity rises to pathological levels , producing EEG traces that show large wave and spike patterns not seen in a healthy brain .
Relating these population - level patterns to the computational functions of individual neurons is a major focus of current research in neurophysiology .
Metabolism All vertebrates have a blood -- brain barrier that allows metabolism inside the brain to operate differently from metabolism in other parts of the
body . Glial cells play a major role in brain metabolism by controlling the chemical composition of the fluid that surrounds neurons , including levels of ions
and nutrients . Brain tissue consumes a large amount of energy in proportion to its volume , so large brains place severe metabolic demands on animals . The
need to limit body weight in order , for example , to fly , has apparently led to selection for a reduction of brain size in some species , such as bats .
</item 7>

<item 8>
Brain
Hum Brain Mapp. 16 ( 1 ) : 14 -- 23 . doi : 10.1002 / hbm. 10026 . PMID 11870923 . Jump up ^ Ebert , D. ; Haller , RG. ; Walton , ME . ( Jul 2003 ) . `` Energy
contribution of octanoate to intact rat brain metabolism measured by 13C nuclear magnetic resonance spectroscopy '' . J Neurosci. 23 ( 13 ) : 5928 -- 35 . PMID
12843297 . Jump up ^ Marin - Valencia , I. ; Good , LB. ; Ma , Q. ; Malloy , CR. ; Pascual , JM . ( Feb 2013 ) . `` Heptanoate as a neural fuel : energetic and
neurotransmitter precursors in normal and glucose transporter I - deficient ( G1D ) brain '' . J Cereb Blood Flow Metab. 33 ( 2 ) : 175 -- 82 . doi : 10.1038 /
jcbfm. 2012.151 . PMC 3564188 .
</item 8>

<item 9>
Brain
`` Acetate transport and utilization in the rat brain '' . J Neurochem. 109 Suppl 1 ( Suppl 1 ) : 46 -- 54 . doi : 10.1111 / j. 1471 - 4159.2009. 05895. x . PMC
2722917 . PMID 19393008 . Jump up ^ Soengas , JL ; Aldegunde , M ( 2002 ) . `` Energy metabolism of fish brain '' . Comparative Biochemistry and Physiology B.
131 ( 3 ) : 271 -- 296 . doi : 10.1016 / S1096 - 4959 ( 02 ) 00022 - 2 . PMID 11959012 . ^ Jump up to : Carew , TJ ( 2000 ) . `` Ch. 1 '' . Behavioral
Neurobiology : the Cellular Organization of Natural Behavior . Sinauer Associates . ISBN 978 - 0 - 87893 - 092 - 0 . Jump up ^ Dafny , N. `` Anatomy of the
spinal cord '' .
</item 9>

<item 10>
Brain
In rituals Some archaeological evidence suggests that the mourning rituals of European Neanderthals also involved the consumption of the brain . The Fore people
of Papua New Guinea are known to eat human brains . In funerary rituals , those close to the dead would eat the brain of the deceased to create a sense of
immortality . A prion disease called kuru has been traced to this . For Tanning The brain can be useful to hunters : most animals have enough brain matter for
use in the tanning of their own hides . See also Brain -- computer interface Central nervous system disease List of neuroscience databases Neurological disorder
Neuroplasticity Optogenetics Outline of neuroscience References Jump up ^ Pelvig , DP ; Pakkenberg , H ; Stark , AK ; Pakkenberg , B ( 2008 ) . `` Neocortical
glial cell numbers in human brains '' . Neurobiology of Aging. 29 ( 11 ) : 1754 -- 1762 . doi : 10.1016 / j. neurobiolaging. 2007.04. 013 . PMID 17544173 . Jump
up ^ Yuste , Rafael ; Church , George M.
</item 10>
=== END ITEM id=75334eadf2d9bbd0 ===

=== ITEM id=89e61cc1c5680d52 ===
Gold sentences and facts:
Sentence 0: The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of law by aggrieved citizens .
  Fact 0.0: The provisions relating to the right to property were changed a number of times.
  Fact 0.1: The Forty-Fourth Amendment of 1978 deleted the right to property from the list of fundamental rights.
  Fact 0.2: A new provision, Article 300-A, was added to the constitution.
  Fact 0.3: Article 300-A provides that no person shall be deprived of his property save by authority of law.
  Fact 0.4: If a legislature makes a law depriving a person of his property, the State has no obligation to pay anything as compensation.
  Fact 0.5: An aggrieved person shall have no right to move the court under Article 32 in that case.
  Fact 0.6: The right to property is no longer a fundamental right.
  Fact 0.7: The right to property is still a constitutional right.
  Fact 0.8: If the government appears to have acted unfairly, the action can be challenged in a court of law by aggrieved citizens.

Retrieved items:
<item 1>
Fundamental rights in India
They are covered in Part III ( Articles 12 to 35 ) of Indian constitution . Contents ( hide ) 1 Genesis 2 Significance and characteristics 3 Right to equality 4
Right to freedom 5 Right against exploitation 6 Right to freedom of religion 7 Right to life 8 Cultural and Educational rights 9 Right to constitutional
remedies 10 Right to Privacy 11 Critical analysis 12 Amendments 13 Fundamental rights not sacrosanct 13.1 Validity of Article 31B 13.2 Amendment to Article 31C
13.3 Right to property 13.4 Right to education 14 See also 15 References 16 Footnotes Genesis ( edit ) The development of constitutionally guaranteed
fundamental human rights in India was inspired by historical examples such as England 's Bill of Rights ( 1689 ) , the United States Bill of Rights ( approved
on 17 September 1787 , final ratification on 15 December 1791 ) and France 's Declaration of the Rights of Man ( created during the revolution of 1789 , and
ratified on 26 August 1789 ) .
</item 1>

<item 2>
Fundamental rights in India
The right to constitutional remedies is present for enforcement of Fundamental Rights . The right to privacy is an intrinsic part of Article 21 that protects
life and liberty of the citizens . Fundamental rights for Indians have also been aimed at overturning the inequalities of pre-independence social practices .
Specifically , they have also been used to abolish untouchability and thus prohibit discrimination on the grounds of religion , race , caste , sex , or place of
birth . They also forbid trafficking of human beings and forced labour ( a crime ) . They also protect cultural and educational rights of religious and
linguistic minorities by allowing them to preserve their languages and also establish and administer their own education institutions . They are covered in Part
III ( Articles 12 to 35 ) of Indian constitution .
</item 2>

<item 3>
Fundamental rights in India
These fundamental rights help not only in protection but also the prevention of gross violations of human rights . They emphasise on the fundamental unity of
India by guaranteeing to all citizens the access and use of the same facilities , irrespective of background . Some fundamental rights apply for persons of any
nationality whereas others are available only to the citizens of India . The right to life and personal liberty is available to all people and so is the right
to freedom of religion . On the other hand , freedoms of speech and expression and freedom to reside and settle in any part of the country are reserved to
citizens alone , including non-resident Indian citizens . The right to equality in matters of public employment can not be conferred to overseas citizens of
India . Fundamental rights primarily protect individuals from any arbitrary state actions , but some rights are enforceable against individuals . For instance ,
the Constitution abolishes untouchability and also prohibits begar . These provisions act as a check both on state action as well as the action of private
individuals . However , these rights are not absolute or uncontrolled and are subject to reasonable restrictions as necessary for the protection of general
welfare . They can also be selectively curtailed .
</item 3>

<item 4>
Fundamental rights in India
. The Times of India . Retrieved 8 July 2010 . Jump up ^ `` Court rejects plea to make property a fundamental right '' . The Hindu . Chennai , India . 19
October 2010 . Jump up ^ `` Education is now a fundamental right of every child '' . Hindustan Times . 1 April 2010 . Archived from the original on 9 July 2010
. Retrieved 8 July 2010 . Footnotes ( edit ) Links to related articles ( hide ) Government of India Constitution Constitution Fundamental Rights , Directive
Principles and Fundamental Duties Federalism ( Union List State List Concurrent List ) Executive Union Government Prime Minister ( list ) Cabinet Ministers
President ( list ) Vice President ( list ) Civil Services All India Services ( IAS IFS / IFoS IPS ) Cabinet Secretary Secretaries ( Defence Finance Foreign Home
) State governments Chief Minister ( list ) Chief Secretary Governors ( list ) Legislative Parliament Lok Sabha ( Speaker ) Rajya Sabha ( Chairman ) Vidhan
Sabha Vidhan Parishad Leader of the Opposition Judiciary Supreme Court ( Chief Justice ) High Courts District Courts Retrieved from ``
https://en.wikipedia.org/w/index.
</item 4>

<item 5>
Fundamental rights in India
Jump up ^ Sovereign means possessing supreme or ultimate power . `` in modern democracies the people 's will is in theory sovereign '' synonyms : supreme ,
absolute , unlimited , unrestricted , unrestrained , unbounded , boundless , infinite , ultimate , total , unconditional , full , utter , paramount ; principal
, chief , dominant , predominant Jump up ^ `` What causes judicial delay ? Judgements diluting time frames in Code of Civil Procedure worsen the problem of
adjournments '' . Retrieved 13 September 2017 . Jump up ^ `` Paras 316 and 317 of Kesavananda Bharati v. State of Kerala , ( AIR 1973 SC 1461 ) '' . 1973 .
Retrieved 21 October 2016 . Jump up ^ `` IX Schedule laws open to review '' . 2007 . Retrieved 21 October 2017 . Jump up ^ 44th Amendment Act , 1978 . Jump up ^
Mahapatra , Dhananjay ( 28 February 2009 ) . `` Should right to property return ? '' . The Times of India . Retrieved 8 July 2010 .
</item 5>

<item 6>
Fundamental rights in India
In 2007 , Supreme Court ruled that there could not be any blanket immunity from judicial review for the laws inserted in the Ninth Schedule . Apex court also
stated it shall examine laws included in the Ninth Schedule after 1973 for any incompatibility with the basic structure doctrine . Amendment to Article 31C (
edit ) Section 4 of the 42nd Amendment , had changed Article 31C of the constitution to accord precedence to the Directive Principles ( earlier applicable only
to clauses b & c of Article 39 ) over the fundamental rights of individuals . In Minerva Mills v. Union of India case , supreme court ruled that the amendment
to the Article 31C is not valid and ultra vires . Right to property ( edit ) The Constitution originally provided for the right to property under Articles 19
and 31 . Article 19 guaranteed to all citizens the right to acquire , hold and dispose of property . Article 31 provided that `` no person shall be deprived of
his property save by authority of law . '' It also provided that compensation would be paid to a person whose property has been taken for public purposes .
</item 6>

<item 7>
Fundamental rights in India
These include individual rights common to most liberal democracies , such as equality before law freedom of speech and expression , religious and cultural
freedom and peaceful assembly , freedom to practice religion , and the right to constitutional remedies for the protection of civil rights by means of writs
such as habeas corpus , Mandamus , Prohibition , Certiorari and Quo Warranto . Violation of these rights result in punishments as prescribed in the Indian Penal
Code or other special laws , subject to discretion of the judiciary . The Fundamental Rights are defined as basic human freedoms that every Indian citizen has
the right to enjoy for a proper and harmonious development of personality . These rights universally apply to all citizens , irrespective of race , place of
birth , religion , caste or gender . Though the rights conferred by the constitution other than fundamental rights are equally valid and their enforcement in
case of violation shall be secured from the judiciary in a time consuming legal process . However , in case of fundamental rights violation , Supreme court of
India can be approached directly for ultimate justice per Article 32 . The Rights have their origins in many sources , including England 's Bill of Rights , the
United States Bill of Rights and France 's Declaration of the Rights of Man .
</item 7>

<item 8>
Fundamental rights in India
Even an unaided minority institution ought not to ignore the merit of the students for admission . Right to constitutional remedies ( edit ) Right to
constitutional remedies ( Article 32 to 35 ) empowers the citizens to move to a court of law in case of any denial of the fundamental rights . For instance , in
case of imprisonment , any citizen can ask the court to see if it is according to the provisions of the law of the country by lodging a PIL . If the court finds
that it is not , the person will have to be freed . This procedure of asking the courts to preserve or safeguard the citizens ' fundamental rights can be done
in various ways . The courts can issue various kinds of writs . These writs are habeas corpus , mandamus , prohibition , quo warranto and certiorari . Dr. B.R.
Ambedkar rightly declared Right to constitutional remedies as `` the heart and soul '' of Indian constitution . When a national or state emergency is declared ,
this right is suspended by the central government . Article 32 is also called citizens right to protect and defend the constitution as it can be used by the
citizens to enforce the constitution through the judiciary .
</item 8>

<item 9>
Fundamental rights in India
The provisions relating to the right to property were changed a number of times . The Forty - Fourth Amendment of 1978 deleted the right to property from the
list of fundamental rights A new provision , Article 300 - A , was added to the constitution , which provided that `` no person shall be deprived of his
property save by authority of law '' . Thus if a legislature makes a law depriving a person of his property , there would be no obligation on the part of the
State to pay anything as compensation . The aggrieved person shall have no right to move the court under Article 32 . Thus , the right to property is no longer
a fundamental right , though it is still a constitutional right . If the government appears to have acted unfairly , the action can be challenged in a court of
law by aggrieved citizens . The liberalisation of the economy and the government 's initiative to set up special economic zones has led to many protests by
farmers and have led to calls for the reinstatement of the fundamental right to private property .
</item 9>

<item 10>
Fundamental rights in India
This is of supreme importance and can not be destroyed by any form of amendment to the constitution . Many constitutional amendments to Part III of the
constitution were made deleting or adding or diluting the fundamental rights before the judgement of Golaknath case ( Constitutional amendments 1 , 4 , 7 and 16
) and after the validity of 24th constitutional amendment is upheld by the Supreme Court ( Constitutional amendments 25 , 42 , 44 , 50 , 77 , 81 , 85 , 86 , 93
and 97 ) . Validity of Article 31B ( edit ) Articles 31A and Article 31B are added by First constitutional amendment in 1951 . Article 31B says that any acts
and regulations included in the Ninth Schedule of the constitution by the Parliament can override the fundamental rights and such laws can not be repealed or
made void by the judiciary on the grounds of violating fundamental rights . Thus fundamental rights given in Part III are not equally applicable in each state /
region and can be made different by making additions / deletions to Ninth Schedule by constitutional amendments .
</item 10>
=== END ITEM id=89e61cc1c5680d52 ===

=== ITEM id=e36d289150b52f23 ===
Gold sentences and facts:
Sentence 0: No . overall No. in season Italian Title / RAI English Title 4Kids English title Italian air date American air date `` Una Fata a Gardenia '' / `` An Unexpected Event '' `` It Feels Like Magic '' 28 January 2004 ( 2004 - 01 - 28 ) 19 June 2004 While on summer vacation , Bloom , an ordinary 16 - year - old high school girl living on Earth , encounters a battle between a 17 - year - old fairy named Stella and an ogre named Knut . As Stella 's magic scepter is stolen , Bloom intervenes , discovering that she has magical powers and manages to turn the tide . Stella invites Bloom to attend Alfea , a school for fairies , to fully master her new powers . Knut returns with more ghouls and a troll , launching a surprise attack at Bloom 's home . Stella and Bloom call in the Specialists , a group of boys from Magix at Red Fountain , a school for boys , who capture the troll and defeat Knut . Sky , Brandon , Timmy , and Riven . The next morning , Bloom has agreed to go to Alfea . `` Benvenuti a Magix ! '' / `` Welcome to Magix ! '' `` More Than High School '' 30 January 2004 ( 2004 - 01 - 30 ) 26 June 2004 Bloom arrives at Alfea , under the name Princess Varanda of Callisto , a student who had declined her admission . She is introduced to her apartment mates , Musa , Tecna , and Flora . The girls then go for a day out in Magix City . While calling her parents , she spots Knut spying on her friends , and follows him . She learns that Knut is working for the Trix : three witches from the school of Cloud Tower named Icy , Darcy , and Stormy . After pitifully trying to defend herself , Bloom is captured by the Trix , but the other girls rescue her . Bloom reveals her real name . When the girls are caught sneaking back into school , Bloom confesses her real name to Headmistress Faragonda and discipline head Griselda , but Faragonda forgives her transgression and lets her attend the school . The girls agree on Bloom 's name for their group : The Winx . `` L'anello di Stella '' / `` Alfea College for Fairies '' `` Save the First Dance '' 2 February 2004 ( 2004 - 02 - 02 ) 3 July 2004 Alfea and the neighboring boys school , Red Fountain , prepare for a ball to begin the school year . However , Headmistress Griffon of Cloudtower plans to spoil the ball as revenge for being excluded . She picks The Trix 's suggestion of replacing the magic egg gifts with snake - rat eggs . While Bloom prepares her dress for the ball , she spots the Trix casting a spell on the eggs , so she tells her friends , and they reverse the spell in time for the ball . Later Bloom spots the Trix trying to steal Stella 's ring , and tries to fight them off , transforming in a fairy for the first time . Although the Trix are able to escape , the other Winx girls inform Bloom that Stella 's ring is safe and that they swapped the contents of the case with a baby duck ( Pepe ) that bonds with Icy . `` La palude di Melmamora '' / `` The Black - Mud Swamp '' `` The Voice of Nature '' 4 February 2004 ( 2004 - 02 - 04 ) 10 July 2004 As part of Professor Palladium 's field trip , the class must navigate the Black - Mud Swamp to reach a clearing in the Gloomy - Wood Forest by listening to nature and without using any magic . The boys from Red Fountain are transporting a troll prisoner but the Trix cause their aircraft to crash land , and the troll is freed . The Winx and the Red Fountain boys end up meeting in the swamp , and must work together , despite their differences , to find their way out . They encounter the troll who is harassing some other Alfea fairies , and chase him off , however , the Trix find the troll and send him away . Although the Red Fountain boys failed to deliver the troll , they remark they are starting to feel like a team . The Winx girls , although they finished last in their assignment , are commended by their teachers for their helping others . 5 5 `` Appuntamento al buio '' / `` Date with Disaster '' `` Date with Disaster '' 6 February 2004 ( 2004 - 02 - 06 ) 17 July 2004 While the Winx are on kitchen duty , Stella is invited on a date with Prince Sky ( Brandon ) at the brand - new Black Lagoon restaurant and also asks for help on picking out an outfit . While selecting her attire , Stella decides not to wear the ring of Solaria and gives it to Bloom for safekeeping ( believing she wo n't need her magic powers ) . The next morning , Stella starts acting completely different , becoming irritable and refusing to say antything about last night . During classes , Bloom notices that Stella is absent and goes back to their dorm to check , only to find Stella has torn the room apart ! An enraged Stella begins asks Bloom where her ring is , and eventually blasts her when she does n't say . Stella then suddenly vanishes , leaving the other four to find Prince Sky at Redfountain for information . To their shock , Sky says he worked out all night and he never wrote the invitation . The Winx then decide to find the Black Lagoon , only it does n't show up on Tecna 's map , and no one in the city seems to know where it is . Just as Musa is ready to quit , a strange man gives them the exact location . The girls find a beautiful restaurant on the edge of town , though Flora finds it strange how the gorgeous tree feels dead . She quickly reveals that an illusion has been cast ; revealing a run - down shack . The Winx venture inside and find Stella who begins attacking them ! The girls transform , and quickly learn that the entire date was a plot by the Trix . They sent the fake invitation and kidnapped her to get her ring of Solaria ! However , when she did n't wear it , they sent Darcy disguised as Stella back to Alfea to find it . The Trix also reveal the real Stella trapped in magic energy rings and force Bloom into an ultimatum ; hand over the ring or Icy will kill Stella . In the end , Bloom decides Stella is more important , transforms out of fairy form and gives them the ring , while also beginning her role as the unofficial leader . 6 6 `` Missione a Torrenuvola '' / `` Mission at Cloudtower '' `` Secret Guardian '' 9 February 2004 ( 2004 - 02 - 09 ) 24 July 2004 After returning to Alfea , Bloom researches the Ring of Solaria , hoping to find out what made the Trix go to such lengths to get it . Meanwhile , the Trix use a spell on the ring , hoping to access an extremely powerful magic known as the Dragon Flame , only to have their spell blow up in their faces as the ring proves negative ! An infuriated Icy throws the ring away and plans to begin the search all over again . Back at Alfea , Bloom learns more on the ring of Solaria , but also learns about the Dragon Flame . Bloom then turns to Faragonda for answers , who tells her that the Dragon Flame was the energy of the Great Dragon , who created the magical universe and later rested on the planet Domino . That night , the Winx plan to infiltrate Cloud Tower to retrieve Stella 's ring ( since she ca n't transform without it ) . Using the same secret tunnel the Trix used earlier , the girls sneak into Cloudtower without a hitch while the Trix are out . After searching their room , Stella finds her ring , but the tunnel does n't reopen and the Winx are forced to find the main exit without getting caught . While exploring , the Winx stumble upon Cloudtower 's archive which contains history on every person in the Magic dimension . A curious Bloom spots a book with her name on it and tries to read more on her past , but quickly regrets it when it turns into a trap that alerts Headmistress Griffin ! The Winx try to make a hasty escape , but Griffin sends Bako monsters to scare them out ! After using their magic to fight the monsters , the Winx are led to a dead end . In a panic to destroy the last monster , Stella accidentally sets the room on fire ! As the Winx begin to suffocate , a mysterious voice guides Bloom to an exit , allowing them to leave Cloudtower . Though the ring was recovered , the Winx are busted since Griffin already called Faragonda hours ago . As punishment , Faragonda temporarily extracts all their powers . 7 7 `` A che servono gli amici ? '' / `` Friends in Need '' `` Grounded '' 11 February 2004 ( 2004 - 02 - 11 ) 31 July 2004 The Winx have lost all their magic as punishment for sneaking into Cloudtower , but to make matters worse , Griselda tasks them with cleaning the entire school without their magic ( or technology , much to Tecna 's dismay ) while the rest of the students and faculty go to a concert . Problems continue as Tecna struggles with no gadgets to help her , causing her to wear a bucket on her head and attempt to use a broom , while Musa , Stella and Bloom have a water fight , wasting their time , energy and supplies . In order to make the punishment lighter , the girls secretly invite the boys from Redfountain over after everyone has left , claiming to be throwing an all - night party . Meanwhile , the Trix use the opportunity to sneak into a near - empty Alfea to find the real source of the Dragon Flame . The boys arrive and instead learn they 'll be working as janitors , though they still play music as they work . Soon after , the Trix teleport to Alfea and summon the vacuums to guide them towards the Dragon Flame , while distracting the gang by summoning a monster . The Winx and specialists quickly meet their distraction , and while the boys try to fight with their weapons , their quickly knocked aside . The Winx , despite having no magic , manage to outsmart the beast by using floor cleaner . Unable to find the Dragon Flame , the Trix depart to Cloudtower , unseen by anyone while the Specialists head back to Redfountain . The next morning , Faragonda sees the damage and hears the story from the Winx and as a reward for defeating an adversary using only wits and resourcefulness , the Winx 's powers are returned to them . 8 8 `` La festa della rosa '' / `` A Friendship Sundered '' `` The Day of the Rose '' 13 February 2004 ( 2004 - 02 - 13 ) 7 August 2004 During the Rose Festival , both Bloom and Riven unwittingly fall into a trap by the Trix to divide the close relationship between the Specialists and the Winx . In the end , Riven thinks that Darcy has saved him in a racing crash and he blames Bloom for what happened then gets mad at his friends for believing Bloom over him and transfers out of the squad of Specialists along with Darcy . 9 9 `` Il tradimento di Riven '' / `` Betrayed ! '' `` Spelled '' 16 February 2004 ( 2004 - 02 - 16 ) 14 August 2004 There are hostilities between the Winx due to Stella 's behavior , and Musa 's feelings for Riven are strained . In a fight with the Trix , the possessor of the Dragon 's Flame is revealed to be Bloom . 10 10 `` La fiamma del drago '' / `` Bloom Tested '' `` Magical Reality Check '' 18 February 2004 ( 2004 - 02 - 18 ) 21 August 2004 It 's time for Bloom 's magical midterm test in a virtual reality machine . The Trix mess with the computer so they can enter it and test her to be sure of Bloom is indeed the one who possesses the Dragon Flame . 11 11 `` Il regno delle ninfe '' / `` The Monster and The Willow '' `` Junior League '' 20 February 2004 ( 2004 - 02 - 20 ) 28 August 2004 Flora experiments with her plants to study for her test . Stella ( who had her clothes torn apart by a plant ) and Tecna ( who was attacked by a plant ) tell Flora to move the plants . She moves them to Black - Mud Swamp . Flora seeks help from the Water Nymphs in finding a special plant . The Water Nymphs tell the Winx about a monster which has been terrorizing them and the Winx decide to fight it . 12 12 `` Miss Magix '' 23 February 2004 ( 2004 - 02 - 23 ) 4 September 2004 On the day before a big test , Stella enters a beauty contest . Lucy from Cloud Tower does too , with help from the Trix , who want to play a bad prank on Lucy by first making her pretty , sabotaging the performances of the other contestants , and then breaking the spell in front of everyone once she has earned the Miss Magix prize . In the end , Stella wins and kept her promise to Bloom , which was to study all night for the test the next morning . Although Stella makes a few mistakes but because she studied after the pageant , she passes . 13 13 `` La figlia del fuoco '' / `` A Great Secret Revealed '' `` Meant to Be '' 25 February 2004 ( 2004 - 02 - 25 ) 11 September 2004 Bloom returns to Earth during spring break . She discovers she can read people 's auras , and she casts her doubts on a pending business deal regarding her mother 's flower shop . A vision was unlocked while putting out a fire in her mother 's flower shop . She also discovers that Mike found her as a baby while tackling a fire , meaning Bloom 's not Mike and Vanessa 's biological daughter . 14 14 `` Il segreto di Bloom '' / `` Bloom 's Dark Secret '' `` Witch Trap '' 27 February 2004 ( 2004 - 02 - 27 ) 18 September 2004 Bloom 's curiosity about her past and magical powers begin to interfere with her confidence and she arranges to go to Cloud Tower with Brandon to seek answers . Bloom meets a kind witch named Mirta . 15 15 `` Voci dal passato '' / `` Honor Above All '' `` Pushing the Envelope '' 1 March 2004 ( 2004 - 03 - 01 ) 25 September 2004 A pending test by Professor Wizgiz worries the Winx Club . Bloom accidentally finds the answers to the test . 16 16 `` Il nemico nell'ombra '' / `` Cold Spell '' `` The Nightmare Monster '' 3 March 2004 ( 2004 - 03 - 03 ) 2 October 2004 In another attempt to harness the power of the Dragon 's Flame , the Trix send a nightmare monster to terrorise the dreams of the Winx . 17 17 `` Il segreto di Brandon '' / `` Secrets Within Secrets '' `` Royal Heartbreak '' 5 March 2004 ( 2004 - 03 - 05 ) 9 October 2004 During festivities at Redfountain , Bloom encounters Princess Diaspro , who claims she is betrothed to Brandon , whom she keeps calling Sky . Bloom finds out that Sky and Brandon switched identities , because `` Sky '' wanted to live a normal life for once . 18 18 `` Addio Magix ! '' / `` The Font of Dragon Fire '' `` Senior Witches Go to Earth '' 8 March 2004 ( 2004 - 03 - 08 ) 16 October 2004 Upset about the events at Redfountain , Bloom returns to Earth . The Trix take advantage of the situation and attack her on Earth . The witches reveal to Bloom that she is the surviving heir to planet Domino ( Sparks ) , and they steal the Dragon Fire from her . 19 19 `` Attacco ad Alfea '' / `` The Fall of Magix '' `` The Army of Decay '' 10 March 2004 ( 2004 - 03 - 10 ) 23 October 2004 Bloom returns to Alfea following the events on Earth . The Trix take control of Cloud Tower and summon the Army of Decay to take control of Magix . 20 20 `` La scomparsa di Bloom '' / `` Mission to Domino '' `` Sparks of Hope '' 12 March 2004 ( 2004 - 03 - 12 ) 30 October 2004 To counter the powerful magic possessed by the Trix , the Winx Club go to planet Domino hoping to find the Dragon 's Flame . Redfountain is attacked by the Army of Decay . 21 21 `` Trappola di ghiaccio '' / `` The Crown of Dreams '' `` The Frozen Palace '' 15 March 2004 ( 2004 - 03 - 15 ) 6 November 2004 In the ruins of Domino ( Sparks ) , the Winx are attacked during their exploration . In Magix , Redfountain is destroyed by the Trix . 22 22 `` Il ritorno di Riven '' / `` Storming Cloudtower '' `` Mission to Cloud Tower '' 17 March 2004 ( 2004 - 03 - 17 ) 13 November 2004 The Army of Decay forces the retreat of the population to Alfea . The Winx go to Cloud Tower in attempt to retrieve the power of the Dragon 's Flame from the Trix . Riven rejoins the Specialists . 23 23 `` Fuga da Torrenuvola '' / `` Power Play '' `` The Search for the Flame '' 19 March 2004 ( 2004 - 03 - 19 ) 20 November 2004 The Trix torment the Winx for being unable to find the power of the Dragon 's Flame on Domino ( Sparks ) . The cadre and staff of Cloud Tower escape to Alfea . 24 24 `` Il mistero del lago '' / `` The Witches ' Siege '' `` Battle for Alfea '' 22 March 2004 ( 2004 - 03 - 22 ) 27 November 2004 The Army of Darkness launches an attack on Alfea . Bloom is guided to Lake Fortress of Light / Roccaluce by a mysterious voice . 25 25 `` Il sonno di Magix '' / `` The Ultimate Challenge '' `` The Great Witch Invasion '' 24 March 2004 ( 2004 - 03 - 24 ) 4 December 2004 The first attack on Alfea fails . Bloom follows the spirit of her older - sister Daphne and regains her Dragon Fire powers . The Trix lead the Army of Decay in a final attack on Alfea . 26 26 `` Battaglia finale '' / `` The Witches ' Downfall '' `` Fire and Ice '' 26 March 2004 ( 2004 - 03 - 26 ) 11 December 2004 As the Army of Darkness approaching Alfea and the Trix seeking to take complete control of the Magix Dimension , the Winx , except Bloom and including Alfea 's students and teachers take action to fend off the Trix from completely destroying algae to the ground before the Trix use the powers of the Dragon . RC1 RC1 `` Attack of the Senior Witches '' N / A 22 May 2004 A clip show episode made by 4Kids Entertainment featuring scenes from episodes 18 , 19 and 20 where the Trix attack Alfea , serving as a sneak preview to the first season .
  Fact 0.0: The table lists episodes with columns for the overall number, the season number, the Italian title / RAI English title, the 4Kids English title, the Italian air date and the American air date.
  Fact 0.1: Episode 5, 'Appuntamento al buio', has the RAI English title and the 4Kids English title 'Date with Disaster'.
  Fact 0.2: Episode 5 aired in Italy on 6 February 2004 and in America on 17 July 2004.
  Fact 0.3: While the Winx are on kitchen duty, Stella is invited on a date with Prince Sky (Brandon) at the brand-new Black Lagoon restaurant.
  Fact 0.4: Stella also asks for help picking out an outfit.
  Fact 0.5: While selecting her attire, Stella decides not to wear the ring of Solaria and gives it to Bloom for safekeeping, believing she won't need her magic powers.
  Fact 0.6: The next morning, Stella starts acting completely differently, becoming irritable and refusing to say anything about the previous night.
  Fact 0.7: During classes, Bloom notices that Stella is absent and goes back to their dorm to check.
  Fact 0.8: Bloom finds that Stella has torn the room apart.
  Fact 0.9: An enraged Stella asks Bloom where her ring is, and eventually blasts her when she does not say.
  Fact 0.10: Stella then suddenly vanishes, leaving the other four to find Prince Sky at Redfountain for information.
  Fact 0.11: Prince Sky says he worked out all night and never wrote the invitation.
  Fact 0.12: The Winx decide to find the Black Lagoon.
  Fact 0.13: The Black Lagoon does not show up on Tecna's map.
  Fact 0.14: No one in the city seems to know where the Black Lagoon is.
  Fact 0.15: Just as Musa is ready to quit, a strange man gives the girls the exact location of the Black Lagoon.
  Fact 0.16: The girls find a beautiful restaurant on the edge of town.
  Fact 0.17: Flora finds it strange how the gorgeous tree feels dead.
  Fact 0.18: An illusion has been cast, revealing a run-down shack.
  Fact 0.19: The Winx find Stella inside the shack, and she begins attacking them.
  Fact 0.20: The girls transform and learn that the entire date was a plot by the Trix.
  Fact 0.21: The Trix sent the fake invitation and kidnapped Stella to get her ring of Solaria.
  Fact 0.22: Because Stella did not wear the ring, the Trix sent Darcy disguised as Stella back to Alfea to find it.
  Fact 0.23: The Trix reveal the real Stella trapped in magic energy rings.
  Fact 0.24: The Trix force Bloom into an ultimatum: hand over the ring or Icy will kill Stella.
  Fact 0.25: Bloom decides Stella is more important, transforms out of fairy form and gives the Trix the ring.
  Fact 0.26: Bloom begins her role as the unofficial leader.
  Fact 0.27: Episode 6, 'Missione a Torrenuvola', has the RAI English title 'Mission at Cloudtower' and the 4Kids English title 'Secret Guardian'.
  Fact 0.28: Episode 6 aired in Italy on 9 February 2004 and in America on 24 July 2004.
  Fact 0.29: After returning to Alfea, Bloom researches the Ring of Solaria, hoping to find out what made the Trix go to such lengths to get it.
  Fact 0.30: The Trix use a spell on the ring, hoping to access an extremely powerful magic known as the Dragon Flame.
  Fact 0.31: The Trix's spell blows up in their faces as the ring proves negative.
  Fact 0.32: An infuriated Icy throws the ring away and plans to begin the search all over again.
  Fact 0.33: Bloom learns more about the Ring of Solaria at Alfea.
  Fact 0.34: Bloom learns about the Dragon Flame at Alfea.
  Fact 0.35: Bloom turns to Faragonda for answers.
  Fact 0.36: Faragonda tells Bloom that the Dragon Flame was the energy of the Great Dragon.
  Fact 0.37: The Great Dragon created the magical universe.
  Fact 0.38: The Great Dragon later rested on the planet Domino.
  Fact 0.39: That night, the Winx plan to infiltrate Cloud Tower to retrieve Stella's ring, since she cannot transform without it.
  Fact 0.40: The Winx use the same secret tunnel the Trix used earlier.
  Fact 0.41: The Winx sneak into Cloudtower without a hitch while the Trix are out.
  Fact 0.42: After searching their room, Stella finds her ring.
  Fact 0.43: The tunnel does not reopen, so the Winx must find the main exit without getting caught.
  Fact 0.44: The Winx stumble upon Cloudtower's archive, which contains history on every person in the Magic dimension.
  Fact 0.45: Bloom spots a book with her name on it and tries to read more about her past.
  Fact 0.46: The book turns into a trap that alerts Headmistress Griffin.
  Fact 0.47: Griffin sends Bako monsters to scare the Winx out.
  Fact 0.48: After using their magic to fight the monsters, the Winx are led to a dead end.
  Fact 0.49: In a panic to destroy the last monster, Stella accidentally sets the room on fire.
  Fact 0.50: As the Winx begin to suffocate, a mysterious voice guides Bloom to an exit, allowing the Winx to leave Cloudtower.
  Fact 0.51: The Winx's ring was recovered.
  Fact 0.52: The Winx are busted since Griffin already called Faragonda hours ago.
  Fact 0.53: As punishment, Faragonda temporarily extracts all the Winx's powers.
  Fact 0.54: Episode 7, 'A che servono gli amici?', has the RAI English title 'Friends in Need' and the 4Kids English title 'Grounded'.
  Fact 0.55: Episode 7 aired in Italy on 11 February 2004 and in America on 31 July 2004.
  Fact 0.56: The Winx have lost all their magic as punishment for sneaking into Cloudtower.
  Fact 0.57: Griselda tasks the Winx with cleaning the entire school without their magic or technology.
  Fact 0.58: The rest of the students and faculty go to a concert while the Winx clean.
  Fact 0.59: Tecna struggles with no gadgets to help her, causing her to wear a bucket on her head and attempt to use a broom.
  Fact 0.60: Musa, Stella and Bloom have a water fight, wasting their time, energy and supplies.
  Fact 0.61: To make the punishment lighter, the girls secretly invite the boys from Redfountain over after everyone has left.
  Fact 0.62: The girls claim they are throwing an all-night party.
  Fact 0.63: The Trix use the opportunity to sneak into a near-empty Alfea to find the real source of the Dragon Flame.
  Fact 0.64: The boys arrive and learn they will be working as janitors.
  Fact 0.65: The boys still play music as they work.
  Fact 0.66: The Trix teleport to Alfea and summon the vacuums to guide them towards the Dragon Flame.
  Fact 0.67: The Trix distract the gang by summoning a monster.
  Fact 0.68: The Winx and the Specialists meet the distraction.
  Fact 0.69: The boys try to fight with their weapons but are quickly knocked aside.
  Fact 0.70: The Winx, despite having no magic, outsmart the beast by using floor cleaner.
  Fact 0.71: The Trix are unable to find the Dragon Flame and depart to Cloudtower, unseen by anyone.
  Fact 0.72: The Specialists head back to Redfountain.
  Fact 0.73: The next morning, Faragonda sees the damage and hears the story from the Winx.
  Fact 0.74: As a reward for defeating an adversary using only wits and resourcefulness, the Winx's powers are returned to them.
  Fact 0.75: Episode 8, 'La festa della rosa', has the RAI English title 'A Friendship Sundered' and the 4Kids English title 'The Day of the Rose'.
  Fact 0.76: Episode 8 aired in Italy on 13 February 2004 and in America on 7 August 2004.
  Fact 0.77: During the Rose Festival, Bloom and Riven unwittingly fall into a trap by the Trix.
  Fact 0.78: The Trix's trap is meant to divide the close relationship between the Specialists and the Winx.
  Fact 0.79: Riven thinks Darcy saved him in a racing crash.
  Fact 0.80: Riven blames Bloom for what happened in the racing crash.
  Fact 0.81: Riven gets mad at his friends for believing Bloom over him.
  Fact 0.82: Riven transfers out of the squad of Specialists along with Darcy.
  Fact 0.83: Episode 9, 'Il tradimento di Riven', has the RAI English title 'Betrayed!' and the 4Kids English title 'Spelled'.
  Fact 0.84: Episode 9 aired in Italy on 16 February 2004 and in America on 14 August 2004.
  Fact 0.85: There are hostilities between the Winx due to Stella's behavior.
  Fact 0.86: Musa's feelings for Riven are strained.
  Fact 0.87: In a fight with the Trix, the possessor of the Dragon's Flame is revealed to be Bloom.
  Fact 0.88: Episode 10, 'La fiamma del drago', has the RAI English title 'Bloom Tested' and the 4Kids English title 'Magical Reality Check'.
  Fact 0.89: Episode 10 aired in Italy on 18 February 2004 and in America on 21 August 2004.
  Fact 0.90: It is time for Bloom's magical midterm test in a virtual reality machine.
  Fact 0.91: The Trix mess with the computer so they can enter it.
  Fact 0.92: The Trix test Bloom to be sure she is indeed the one who possesses the Dragon Flame.
  Fact 0.93: Episode 11, 'Il regno delle ninfe', has the RAI English title 'The Monster and The Willow' and the 4Kids English title 'Junior League'.
  Fact 0.94: Episode 11 aired in Italy on 20 February 2004 and in America on 28 August 2004.
  Fact 0.95: Flora experiments with her plants to study for her test.
  Fact 0.96: Stella, whose clothes were torn apart by a plant, tells Flora to move the plants.
  Fact 0.97: Tecna, who was attacked by a plant, tells Flora to move the plants.
  Fact 0.98: Flora moves the plants to Black-Mud Swamp.
  Fact 0.99: Flora seeks help from the Water Nymphs in finding a special plant.
  Fact 0.100: The Water Nymphs tell the Winx about a monster that has been terrorizing them.
  Fact 0.101: The Winx decide to fight the monster.
  Fact 0.102: Episode 12, 'Miss Magix', aired in Italy on 23 February 2004 and in America on 4 September 2004.
  Fact 0.103: On the day before a big test, Stella enters a beauty contest.
  Fact 0.104: Lucy from Cloud Tower enters the beauty contest too, with help from the Trix.
  Fact 0.105: The Trix want to play a bad prank on Lucy by first making her pretty.
  Fact 0.106: The Trix plan to sabotage the performances of the other contestants.
  Fact 0.107: The Trix plan to break the spell on Lucy in front of everyone once she has earned the Miss Magix prize.
  Fact 0.108: In the end, Stella wins and keeps her promise to Bloom to study all night for the test the next morning.
  Fact 0.109: Stella makes a few mistakes on the test.
  Fact 0.110: Because Stella studied after the pageant, she passes the test.
  Fact 0.111: Episode 13, 'La figlia del fuoco', has the RAI English title 'A Great Secret Revealed' and the 4Kids English title 'Meant to Be'.
  Fact 0.112: Episode 13 aired in Italy on 25 February 2004 and in America on 11 September 2004.
  Fact 0.113: Bloom returns to Earth during spring break.
  Fact 0.114: Bloom discovers she can read people's auras.
  Fact 0.115: Bloom casts doubts on a pending business deal regarding her mother's flower shop.
  Fact 0.116: A vision was unlocked while Bloom was putting out a fire in her mother's flower shop.
  Fact 0.117: Bloom discovers that Mike found her as a baby while tackling a fire.
  Fact 0.118: Bloom is not Mike and Vanessa's biological daughter.
  Fact 0.119: Episode 14, 'Il segreto di Bloom', has the RAI English title 'Bloom's Dark Secret' and the 4Kids English title 'Witch Trap'.
  Fact 0.120: Episode 14 aired in Italy on 27 February 2004 and in America on 18 September 2004.
  Fact 0.121: Bloom's curiosity about her past and magical powers begins to interfere with her confidence.
  Fact 0.122: Bloom arranges to go to Cloud Tower with Brandon to seek answers.
  Fact 0.123: Bloom meets a kind witch named Mirta.
  Fact 0.124: Episode 15, 'Voci dal passato', has the RAI English title 'Honor Above All' and the 4Kids English title 'Pushing the Envelope'.
  Fact 0.125: Episode 15 aired in Italy on 1 March 2004 and in America on 25 September 2004.
  Fact 0.126: A pending test by Professor Wizgiz worries the Winx Club.
  Fact 0.127: Bloom accidentally finds the answers to the test.
  Fact 0.128: Episode 16, 'Il nemico nell'ombra', has the RAI English title 'Cold Spell' and the 4Kids English title 'The Nightmare Monster'.
  Fact 0.129: Episode 16 aired in Italy on 3 March 2004 and in America on 2 October 2004.
  Fact 0.130: In another attempt to harness the power of the Dragon's Flame, the Trix send a nightmare monster.
  Fact 0.131: The nightmare monster terrorizes the dreams of the Winx.
  Fact 0.132: Episode 17, 'Il segreto di Brandon', has the RAI English title 'Secrets Within Secrets' and the 4Kids English title 'Royal Heartbreak'.
  Fact 0.133: Episode 17 aired in Italy on 5 March 2004 and in America on 9 October 2004.
  Fact 0.134: During festivities at Redfountain, Bloom encounters Princess Diaspro.
  Fact 0.135: Princess Diaspro claims she is betrothed to Brandon, whom she keeps calling Sky.
  Fact 0.136: Bloom finds out that Sky and Brandon switched identities.
  Fact 0.137: Sky and Brandon switched identities because 'Sky' wanted to live a normal life for once.
  Fact 0.138: Episode 18, 'Addio Magix!', has the RAI English title 'The Font of Dragon Fire' and the 4Kids English title 'Senior Witches Go to Earth'.
  Fact 0.139: Episode 18 aired in Italy on 8 March 2004 and in America on 16 October 2004.
  Fact 0.140: Upset about the events at Redfountain, Bloom returns to Earth.
  Fact 0.141: The Trix take advantage of the situation and attack Bloom on Earth.
  Fact 0.142: The witches reveal to Bloom that she is the surviving heir to planet Domino (Sparks).
  Fact 0.143: The witches steal the Dragon Fire from Bloom.
  Fact 0.144: Episode 19, 'Attacco ad Alfea', has the RAI English title 'The Fall of Magix' and the 4Kids English title 'The Army of Decay'.
  Fact 0.145: Episode 19 aired in Italy on 10 March 2004 and in America on 23 October 2004.
  Fact 0.146: Bloom returns to Alfea following the events on Earth.
  Fact 0.147: The Trix take control of Cloud Tower.
  Fact 0.148: The Trix summon the Army of Decay to take control of Magix.
  Fact 0.149: Episode 20, 'La scomparsa di Bloom', has the RAI English title 'Mission to Domino' and the 4Kids English title 'Sparks of Hope'.
  Fact 0.150: Episode 20 aired in Italy on 12 March 2004 and in America on 30 October 2004.
  Fact 0.151: To counter the powerful magic possessed by the Trix, the Winx Club go to planet Domino.
  Fact 0.152: The Winx Club hope to find the Dragon's Flame on planet Domino.
  Fact 0.153: Redfountain is attacked by the Army of Decay.
  Fact 0.154: Episode 21, 'Trappola di ghiaccio', has the RAI English title 'The Crown of Dreams' and the 4Kids English title 'The Frozen Palace'.
  Fact 0.155: Episode 21 aired in Italy on 15 March 2004 and in America on 6 November 2004.
  Fact 0.156: In the ruins of Domino (Sparks), the Winx are attacked during their exploration.
  Fact 0.157: In Magix, the Trix destroy Redfountain.
  Fact 0.158: Episode 22, 'Il ritorno di Riven', has the RAI English title 'Storming Cloudtower' and the 4Kids English title 'Mission to Cloud Tower'.
  Fact 0.159: Episode 22 aired in Italy on 17 March 2004 and in America on 13 November 2004.
  Fact 0.160: The Army of Decay forces the retreat of the population to Alfea.
  Fact 0.161: The Winx go to Cloud Tower to retrieve the power of the Dragon's Flame from the Trix.
  Fact 0.162: Riven rejoins the Specialists.
  Fact 0.163: Episode 23, 'Fuga da Torrenuvola', has the RAI English title 'Power Play' and the 4Kids English title 'The Search for the Flame'.
  Fact 0.164: Episode 23 aired in Italy on 19 March 2004 and in America on 20 November 2004.
  Fact 0.165: The Trix torment the Winx for being unable to find the power of the Dragon's Flame on Domino (Sparks).
  Fact 0.166: The cadre and staff of Cloud Tower escape to Alfea.
  Fact 0.167: Episode 24, 'Il mistero del lago', has the RAI English title 'The Witches' Siege' and the 4Kids English title 'Battle for Alfea'.
  Fact 0.168: Episode 24 aired in Italy on 22 March 2004 and in America on 27 November 2004.
  Fact 0.169: The Army of Darkness launches an attack on Alfea.
  Fact 0.170: Bloom is guided to Lake Fortress of Light / Roccaluce by a mysterious voice.
  Fact 0.171: Episode 25, 'Il sonno di Magix', has the RAI English title 'The Ultimate Challenge' and the 4Kids English title 'The Great Witch Invasion'.
  Fact 0.172: Episode 25 aired in Italy on 24 March 2004 and in America on 4 December 2004.
  Fact 0.173: The first attack on Alfea fails.
  Fact 0.174: Bloom follows the spirit of her older sister Daphne.
  Fact 0.175: Bloom regains her Dragon Fire powers.
  Fact 0.176: The Trix lead the Army of Decay in a final attack on Alfea.
  Fact 0.177: Episode 26, 'Battaglia finale', has the RAI English title 'The Witches' Downfall' and the 4Kids English title 'Fire and Ice'.
  Fact 0.178: Episode 26 aired in Italy on 26 March 2004 and in America on 11 December 2004.
  Fact 0.179: The Army of Darkness approaches Alfea.
  Fact 0.180: The Trix seek to take complete control of the Magix Dimension.
  Fact 0.181: The Winx, except Bloom, take action to fend off the Trix.
  Fact 0.182: Alfea's students and teachers take action to fend off the Trix.
  Fact 0.183: They fend off the Trix to keep the Trix from completely destroying algae to the ground before the Trix use the powers of the Dragon.
  Fact 0.184: RC1 is the Italian title and the RAI English title of the next episode.
  Fact 0.185: The 4Kids English title of RC1 is 'Attack of the Senior Witches'.
  Fact 0.186: RC1 has no Italian air date.
  Fact 0.187: RC1 aired in America on 22 May 2004.
  Fact 0.188: RC1 is a clip show episode made by 4Kids Entertainment.
  Fact 0.189: RC1 features scenes from episodes 18, 19 and 20.
  Fact 0.190: The scenes in RC1 show the Trix attacking Alfea.
  Fact 0.191: RC1 serves as a sneak preview to the first season.

Retrieved items:
<item 1>
Bloom’s attempt to investigate her past at Cloudtower triggers a trap, but a mysterious voice helps the Winx escape. Faragonda punishes them by removing their
powers. While powerless, they clean Alfea, outwit a monster with floor cleaner, and earn their magic back; meanwhile, the Trix search for the Dragon Flame. The
Trix later manipulate Bloom and Riven, straining friendships and driving Riven from the Specialists, though he eventually returns. Bloom is revealed as the
</item 1>

<item 2>
Winx Club (season 1)
Bloom reveals her real name . When the girls are caught sneaking back into school , Bloom confesses her real name to Headmistress Faragonda and discipline head
Griselda , but Faragonda forgives her transgression and lets her attend the school . The girls agree on Bloom 's name for their group : The Winx . `` L'anello
di Stella '' / `` Alfea College for Fairies '' `` Save the First Dance '' 2 February 2004 ( 2004 - 02 - 02 ) 3 July 2004 Alfea and the neighboring boys school ,
Red Fountain , prepare for a ball to begin the school year . However , Headmistress Griffon of Cloudtower plans to spoil the ball as revenge for being excluded
. She picks The Trix 's suggestion of replacing the magic egg gifts with snake - rat eggs . While Bloom prepares her dress for the ball , she spots the Trix
casting a spell on the eggs , so she tells her friends , and they reverse the spell in time for the ball .
</item 2>

<item 3>
Bloom reveals her real name to Faragonda, who allows her to stay; the girls name their group the Winx. The Trix repeatedly sabotage school life and test Bloom’s
connection to the Dragon Flame. Bloom discovers Mike found her as a baby during a fire, learns she is Domino’s surviving heir, and seeks answers at Cloud Tower.
She also learns Sky and Brandon have switched identities. The Trix steal Bloom’s Dragon Flame, seize Cloud Tower, and unleash the Army of Decay.
</item 3>

<item 4>
Winx Club (season 1)
A vision was unlocked while putting out a fire in her mother 's flower shop . She also discovers that Mike found her as a baby while tackling a fire , meaning
Bloom 's not Mike and Vanessa 's biological daughter . 14 14 `` Il segreto di Bloom '' / `` Bloom 's Dark Secret '' `` Witch Trap '' 27 February 2004 ( 2004 -
02 - 27 ) 18 September 2004 Bloom 's curiosity about her past and magical powers begin to interfere with her confidence and she arranges to go to Cloud Tower
with Brandon to seek answers . Bloom meets a kind witch named Mirta . 15 15 `` Voci dal passato '' / `` Honor Above All '' `` Pushing the Envelope '' 1 March
2004 ( 2004 - 03 - 01 ) 25 September 2004 A pending test by Professor Wizgiz worries the Winx Club . Bloom accidentally finds the answers to the test .
</item 4>

<item 5>
Winx Club (season 1)
Stella and Bloom call in the Specialists , a group of boys from Magix at Red Fountain , a school for boys , who capture the troll and defeat Knut . Sky ,
Brandon , Timmy , and Riven . The next morning , Bloom has agreed to go to Alfea . `` Benvenuti a Magix ! '' / `` Welcome to Magix ! '' `` More Than High School
'' 30 January 2004 ( 2004 - 01 - 30 ) 26 June 2004 Bloom arrives at Alfea , under the name Princess Varanda of Callisto , a student who had declined her
admission . She is introduced to her apartment mates , Musa , Tecna , and Flora . The girls then go for a day out in Magix City . While calling her parents ,
she spots Knut spying on her friends , and follows him . She learns that Knut is working for the Trix : three witches from the school of Cloud Tower named Icy ,
Darcy , and Stormy . After pitifully trying to defend herself , Bloom is captured by the Trix , but the other girls rescue her .
</item 5>

<item 6>
Bloom joins Alfea under a false identity, befriends Musa, Tecna, and Flora, and discovers the Trix—Cloud Tower witches Icy, Darcy, and Stormy—are behind Knut
and the troll. She first transforms while protecting Stella’s Ring of Solaria. The Winx and Red Fountain Specialists cooperate to escape a swamp and confront
the troll. The Trix trick and kidnap Stella to steal her ring; Bloom gives it up to save her, emerging as
</item 6>

<item 7>
Winx Club (season 1)
During classes , Bloom notices that Stella is absent and goes back to their dorm to check , only to find Stella has torn the room apart ! An enraged Stella
begins asks Bloom where her ring is , and eventually blasts her when she does n't say . Stella then suddenly vanishes , leaving the other four to find Prince
Sky at Redfountain for information . To their shock , Sky says he worked out all night and he never wrote the invitation . The Winx then decide to find the
Black Lagoon , only it does n't show up on Tecna 's map , and no one in the city seems to know where it is . Just as Musa is ready to quit , a strange man gives
them the exact location . The girls find a beautiful restaurant on the edge of town , though Flora finds it strange how the gorgeous tree feels dead . She
quickly reveals that an illusion has been cast ; revealing a run - down shack . The Winx venture inside and find Stella who begins attacking them ! The girls
transform , and quickly learn that the entire date was a plot by the Trix .
</item 7>

<item 8>
Winx Club (season 1)
The girls transform , and quickly learn that the entire date was a plot by the Trix . They sent the fake invitation and kidnapped her to get her ring of Solaria
! However , when she did n't wear it , they sent Darcy disguised as Stella back to Alfea to find it . The Trix also reveal the real Stella trapped in magic
energy rings and force Bloom into an ultimatum ; hand over the ring or Icy will kill Stella . In the end , Bloom decides Stella is more important , transforms
out of fairy form and gives them the ring , while also beginning her role as the unofficial leader . 6 6 `` Missione a Torrenuvola '' / `` Mission at Cloudtower
'' `` Secret Guardian '' 9 February 2004 ( 2004 - 02 - 09 ) 24 July 2004 After returning to Alfea , Bloom researches the Ring of Solaria , hoping to find out
what made the Trix go to such lengths to get it .
</item 8>

<item 9>
Winx Club (season 1)
12 12 `` Miss Magix '' 23 February 2004 ( 2004 - 02 - 23 ) 4 September 2004 On the day before a big test , Stella enters a beauty contest . Lucy from Cloud
Tower does too , with help from the Trix , who want to play a bad prank on Lucy by first making her pretty , sabotaging the performances of the other
contestants , and then breaking the spell in front of everyone once she has earned the Miss Magix prize . In the end , Stella wins and kept her promise to Bloom
, which was to study all night for the test the next morning . Although Stella makes a few mistakes but because she studied after the pageant , she passes . 13
13 `` La figlia del fuoco '' / `` A Great Secret Revealed '' `` Meant to Be '' 25 February 2004 ( 2004 - 02 - 25 ) 11 September 2004 Bloom returns to Earth
during spring break . She discovers she can read people 's auras , and she casts her doubts on a pending business deal regarding her mother 's flower shop .
</item 9>

<item 10>
Winx Club (season 1)
Bloom accidentally finds the answers to the test . 16 16 `` Il nemico nell'ombra '' / `` Cold Spell '' `` The Nightmare Monster '' 3 March 2004 ( 2004 - 03 - 03
) 2 October 2004 In another attempt to harness the power of the Dragon 's Flame , the Trix send a nightmare monster to terrorise the dreams of the Winx . 17 17
`` Il segreto di Brandon '' / `` Secrets Within Secrets '' `` Royal Heartbreak '' 5 March 2004 ( 2004 - 03 - 05 ) 9 October 2004 During festivities at
Redfountain , Bloom encounters Princess Diaspro , who claims she is betrothed to Brandon , whom she keeps calling Sky . Bloom finds out that Sky and Brandon
switched identities , because `` Sky '' wanted to live a normal life for once . 18 18 `` Addio Magix ! ''
</item 10>
=== END ITEM id=e36d289150b52f23 ===

=== ITEM id=e2235d5f9161cac1 ===
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
Gina Tognoni (born November 28, 1973, in St. Louis) is an American actress and businesswoman known for daytime soap operas. She debuted as Kelly Cramer on *One
Life to Live* (1995), later played Dinah Marler on *Guiding Light* (2004–09), and joined *The Young and the Restless* as Phyllis Summers in 2014. She won two
Daytime Emmys for supporting actress (
</item 1>

<item 2>
Sheila Carter
CBS.com . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Wednesday , 11th January 2006 '' . CBS.com . Retrieved 16 September 2012 .
Jump up ^ `` The Young and the Restless : Thursday , 28th December 2006 '' . CBS.com . Retrieved 16 September 2012 . Jump up ^ `` Michelle Stafford Playing
Sheila Carter on The Young and the Restless '' . She Knows Soaps . Retrieved 16 September 2012 . Jump up ^ Kimberlin Brown ( 2007 ) . KB Interview Nov 2007 .
KOUS News . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday , 9th February 2007 '' . CBS.com . Retrieved 16 September 2012 . Jump
up ^ Kerr , Luke ( 28 August 2009 ) . `` Could Kimberlin Brown Return to B&B ? '' . Daytime Confidential . Retrieved 16 September 2012 .
</item 2>

<item 3>
Gina Tognoni
She is currently starring as Phyllis Summers on The Young and the Restless . Contents ( hide ) 1 Career 2 Personal life 3 Filmography 4 Awards and nominations 5
References 6 External links Career ( edit ) Tognoni is known for her roles on daytime television soap operas . In 1995 , she made her soap debut as Kelly Cramer
in ABC 's One Life to Live . She appeared on the show until 2001 . In April 2001 , it was announced that Tognoni , after six years with the show , had decided
not to renew her contract with the series . Tognoni returned to daytime television as Dinah Marler on CBS 's Guiding Light in July 2004 , and won her first
Daytime Emmy Award for Outstanding Supporting Actress in a Drama Series in 2006 for her work on the show . In 2007 , she was nominated again in the same
category but lost to Genie Francis . She won her second Emmy in 2008 , becoming only the second actress to be a repeat winner in the Supporting Actress category
.
</item 3>

<item 4>
Sheila Carter is a continuing character on *The Young and the Restless*, first appearing in May 1990. Kimberlin Brown has played her since 1990, with Michelle
Stafford portraying her in 2006–07. Sheila has appeared in several stints and crossed over to *The Bold and the Beautiful*. Known by numerous aliases, she has
worked as a waitress, executive assistant, and nurse. Her family includes mother Molly, sister Sarah Smythe, and husband Scott Grainger Sr
</item 4>

<item 5>
Sheila Carter
She then uses crutches to walk , has a seat and asks him for a new face . Months later : it is revealed that Sheila Carter is still alive , had plastic surgery
to look like Phyllis Summers Newman ( Michelle Stafford ) , and that Paul Williams had been hiding the fact that he had been keeping her locked in a cage she
had constructed in an abandoned warehouse . Ultimately , Sheila is determined to escape and take Phyllis 's place in order to infiltrate Michael and Lauren 's
lives . Sheila is accidentally freed from the prison she created for Lauren by Maggie Sullivan , Paul 's girlfriend . After acting as a kidnap victim and
pleading for Maggie 's help , Maggie sets her free but Sheila chokes her unconscious , steals her gun , and then shoots her , but Maggie survives . Subsequently
, Paul returns to the warehouse only to be ambushed by Sheila who imprisons both Paul and Maggie in that cage . Sheila then went to Phyllis ' , tied her up and
waited for Lauren and baby Fen to arrive while taunting Phyllis about how great it would be to take over her life with her husband Nicholas Newman ( Joshua
Morrow ) and her baby Summer .
</item 5>

<item 6>
Sheila Carter
Despite being a regular cast member on The Bold and the Beautiful , Brown returned to The Young and the Restless for several guest appearances beginning on
October 27 , 1992 . Brown departed on November 5 , 1992 , and reappeared for one episode on December 23 , 1992 . She continued making appearance from January 7
to 14 , 1993 , February 19 , 1993 , to March 2 , 1993 , October 11 , 1993 , October 19 to 27 , 1993 , January 12 , 1994 , and February 28 , 1995 , until March 2
, 1995 . In July 2005 , after a 10 - year absence from The Young and the Restless , it was announced that Brown was to reprise the role . At the time , Brown
had been appearing on ABC 's One Life to Live as Paige Miller , and Brown reportedly opted to leave the role when ABC was unable to match the financial terms of
a contract offered to her by CBS . She returned on August 5 , 2005 , and her run ended on January 11 , 2006 .
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
( 1991 -- 92 ) Eric Forrester ( 1993 -- 95 ) James Warwick ( 1997 -- 98 ) Sons Ryder Callahan Daughters Mary Carter Warwick Diana Carter Daisy Carter
Granddaughters Lucy Romalotti Sheila Carter is a fictional character from The Young and the Restless and The Bold and the Beautiful , American soap operas on
the CBS network . Created by William J. Bell , the role was introduced in 1990 -- by Edward J. Scott -- under the portrayal of Kimberlin Brown , who portrayed
the role for many intervals on both soaps until 2006 . That same year , Michelle Stafford took over the role , after Sheila had plastic surgery to look like
Phyllis Summers . Brown returned to the role of Sheila on The Bold and the Beautiful , beginning on June 9 , 2017 . Sheila is known as a villain . A significant
portion of Sheila 's history on both soaps revolves around her long - running conflict with Lauren Fenmore , whom she has attempted to kill on multiple
occasions .
</item 8>

<item 9>
Sheila Carter
Sheila continues to insinuate herself into Eric 's life , faking a concussion after a catfight with Quinn to manipulate Eric into letting her stay in the
Forrester mansion . Sheila blackmails her ex-husband , James Warwick , into lying about the severity of her condition . Critical reception ( edit ) Sheila is
often considered to be one of the most popular villains in daytime . Kimberlin Brown was nominated for a Daytime Emmy in the role as Outstanding Supporting
Actress in 1993 . With the global success of The Bold and the Beautiful , Sheila is arguably the most recognisable soap opera villain in the world . References
( edit ) ^ Jump up to : `` About Y&R : Who 's Who in Los Angeles - Sheila Carter '' . SoapCental . Retrieved 16 September 2012 . ^ Jump up to : Kroll , Dan J. (
9 July 2005 ) . `` Y&R confirms the return of Kimberlin Brown '' . SoapCentral . Retrieved 16 September 2012 . Jump up ^ `` The Young and the Restless : Friday
, 5th August 2005 '' . CBS.com .
</item 9>

<item 10>
Sheila Carter is a villain in CBS’s *The Young and the Restless* and *The Bold and the Beautiful*, created by William J. Bell and introduced in 1990. Kimberlin
Brown played her in multiple runs through 2006; Michelle Stafford took over after Sheila’s plastic surgery made her resemble Phyllis Summers. Brown returned in
2017. Sheila’s long-running feud with Lauren Fenmore includes repeated attempts to kill her; she has also clashed with several Forrester
</item 10>
=== END ITEM id=e2235d5f9161cac1 ===

=== ITEM id=9ede402420143753 ===
Gold sentences and facts:
Sentence 0: A subcutaneous injection is administered as a bolus into the subcutis , the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut is the preferred abbreviation for patient safety .
  Fact 0.0: A subcutaneous injection is administered as a bolus into the subcutis.
  Fact 0.1: The subcutis is the layer of skin directly below the dermis and epidermis.
  Fact 0.2: The dermis and epidermis are collectively referred to as the cutis.
  Fact 0.3: Subcutaneous injections are highly effective in administering vaccines.
  Fact 0.4: Subcutaneous injections are highly effective in administering medications such as insulin, morphine, diacetylmorphine and goserelin.
  Fact 0.5: Subcutaneous injection of recreational drugs, as opposed to intravenous injection, is referred to as 'skin popping'.
  Fact 0.6: Subcutaneous administration may be abbreviated as SC, SQ, sub-cu, sub-Q, SubQ, or subcut.
  Fact 0.7: Subcut is the preferred abbreviation for patient safety.

Retrieved items:
<item 1>
Subcutaneous injection
musculoskeletal Intravenous Intracardiac Intramuscular Intraosseous Intraperitoneal Nanocell injection Patient - Controlled Analgesia pump PIC line WikiProject
Pharmacy portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Subcutaneous_injection&oldid=827949994 '' Categories : Dosage forms Routes of
administration Injection ( medicine ) Talk Contents About Wikipedia Dansk Deutsch Español فارسی Français Italiano Nederlands Norsk Polski Português Slovenščina
Türkçe Українська Tiếng Việt 6 more Edit links This page was last edited on 27 February 2018 , at 18 : 16 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 1>

<item 2>
Glenoid cavity
edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online , SUNY Downstate Medical Center Mechanics of Glenohumeral Instability at University of Washington
Department of Orthopaedics ( hide ) Bones of the arm Shoulder girdle , clavicle conoid tubercle trapezoid line costal tuberosity subclavian groove Scapula
fossae ( subscapular , supraspinatous , infraspinatous ) notches ( suprascapular , great scapular ) glenoid cavity tubercles ( infraglenoid , supraglenoid )
spine of scapula acromion coracoid process angles ( superior , inferior , lateral ) Humerus upper extremity : necks ( anatomical , surgical ) tubercles (
greater , lesser ) bicipital groove body : radial sulcus deltoid tuberosity lower extremity : capitulum trochlea epicondyles ( lateral , medial ) supracondylar
ridges ( lateral , medial ) fossae ( radial , coronoid , olecranon ) Forearm Radius near elbow ( head , tuberosity ) near wrist ( ulnar notch , styloid process
,
</item 2>

<item 3>
Subcutaneous injection
It is slower than intramuscular injections but still faster than intradermal injections . Contents ( hide ) 1 Injection sites 2 Equipment 3 Procedure 4
Subcutaneous insulin injections in diabetes mellitus 5 See also 6 References Injection sites ( edit ) Subcutaneous injection sites Sites include , The outer
area of the upper arm . The abdomen , from the rib margin to the iliac crest and avoiding a 2 - inch circle around the navel . This has the fastest rate of
absorption among the sites . The front of the thigh , midway to the outer side , 4 inches below the top of the thigh to 4 inches above the knee . This has a
slower rate of absorption than the upper arm . The upper back . The upper area of the buttock , just behind the hip bone . This has the slowest rate of
absorption among the sites . Equipment ( edit ) A 25 to 31 gauge thick , 3 / 8 '' to 1 '' long needle can be used . The size is determined by the amount of
subcutaneous tissue present , which is based on patient build .
</item 3>

<item 4>
Glenoid cavity
Glenoid cavity - wikipedia Glenoid cavity Jump to : navigation , search Glenoid cavity Costal surface of left scapula . Glenoid cavity shown in red . Glenoid
cavity shown in red . This cavity articulates with the head of the humerus . Details Identifiers Latin cavitas glenoidalis , fossa glenoidalis Dorlands /
Elsevier c_16 / 12220465 TA A02. 4.01. 019 FMA 23275 Anatomical terms of bone ( edit on Wikidata ) The glenoid cavity or glenoid fossa of scapula is a part of
the shoulder . It is a shallow , pyriform articular surface , which is located on the lateral angle of the scapula . It is directed laterally and forward and
articulates with the head of the humerus ; it is broader below than above and its vertical diameter is the longest . This cavity forms the glenohumeral joint
along with the humerus . This type of joint is classified as a synovial , ball and socket joint .
</item 4>

<item 5>
Subcutaneous injection
25.2007. s112 . ( hide ) Routes of administration , dosage forms Oral Digestive tract ( enteral ) Solids Pill Tablet Capsule Pastille Time release technology
Osmotic delivery system ( OROS ) Liquids Decoction Elixir Electuary Emulsion Extended - release syrup Effervescent powder or tablet Herbal tea Hydrogel
Molecular encapsulation Powder Softgel Solution Suspension Syrup Syrup Concentrate for dilution and / or addition of carbonated water Tincture Buccal (
sublabial ) , sublingual Solids Orally disintegrating tablet ( ODT ) Film Lollipop Sublingual drops Lozenges Effervescent buccal tablet Chewing gum Liquids
Mouthwash Toothpaste Ointment Oral spray Respiratory tract Solids Smoking device Dry - powder inhaler ( DPI ) 0 0 Liquids Anaesthetic vaporizer Vaporizer
Nebulizer Metered - dose inhaler ( MDI ) Gas Oxygen mask and Nasal cannula Oxygen concentrator Anaesthetic machine Relative analgesia machine Ophthalmic ,
otologic ,
</item 5>

<item 6>
Serratus anterior muscle
The muscle is inserted along the medial border of the scapula between the superior and inferior angles along with being inserted along the thoracic vertebrae .
The muscle is divided into three named parts depending on their points of insertions : the serratus anterior superior is inserted near the superior angle the
serratus anterior intermediate is inserted along the medial border the serratus anterior inferior is inserted near the inferior angle . Relations ( edit ) The
serratus anterior lies deep to the subscapularis , from which it is separated by the subscapularis ( supraserratus ) bursa . It is separated from the rib by the
scapulothoracic ( infraserratus ) bursa . Nerve supply ( edit ) The serratus anterior is innervated by the long thoracic nerve ( Nerve of Bell ) , a branch of
the brachial plexus . The long thoracic nerve travels inferiorly on the surface of the serratus . The nerve is especially vulnerable during certain types of
surgery ( for example , during lymph node clearance from the axilla ( e.g.
</item 6>

<item 7>
Subcutaneous injection
The 3 / 8 '' and 5 / 8 '' needles are most commonly used . Usually , no more than 1 mL of solution is given , compared to intradermal injections , where no more
than 0.5 mL is usually given . Procedure ( edit ) Subcutaneous injections are inserted at 45 to 90 degree angles , depending on amount of subcutaneous tissue
present and length of needle - a shorter , 3 / 8 '' needle is usually inserted 90 degrees and a 5 / 8 '' needle is usually inserted at 45 degrees . Medication
is administered slowly , about 10 seconds / milliliter . Subcutaneous insulin injections in diabetes mellitus ( edit ) An insulin pump with a subcutaneous
injection site . A person with insulin - dependent ( either type 1 or type 2 ) diabetes mellitus typically injects insulin subcutaneously . The injection should
be given under the skin , into the fat layer .
</item 7>

<item 8>
Subcutaneous injection
otologic , nasal Nasal spray Ear drops Eye drops Ointment Hydrogel Nanosphere suspension Insufflation Mucoadhesive microdisc ( microsphere tablet ) Urogenital
Ointment Pessary ( vaginal suppository ) Vaginal ring Vaginal douche Intrauterine device ( IUD ) Extra-amniotic infusion Intravesical infusion Rectal ( enteral
) Ointment Suppository Enema Solution Hydrogel Murphy drip Nutrient enema Dermal Ointment Topical cream Topical gel Liniment Paste Film DMSO drug solution
Electrophoretic dermal delivery system Hydrogel Liposomes Transfersome vesicles Cream Lotion Lip balm Medicated shampoo Dermal patch Transdermal patch Contact (
rubbed into break in the skin ) Transdermal spray Jet injector Injection , infusion ( into tissue / blood ) Skin Intradermal Subcutaneous Transdermal implant
Organs Intracavernous Intravitreal Intra-articular injection Transscleral Central nervous system Intracerebral Intrathecal Epidural Circulatory ,
</item 8>

<item 9>
Subcutaneous injection
Subcutaneous injection - wikipedia Subcutaneous injection Jump to : navigation , search `` Subcut '' redirects here . For the song by Grand Mixer DXT featuring
Bill Laswell , see Aftermathematics . Play media 3D animation of a subcutaneous injection A subcutaneous injection is administered as a bolus into the subcutis
, the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in
administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of
recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut
is the preferred abbreviation for patient safety . Subcutaneous tissue has few blood vessels and so drugs injected here are for slow , sustained rates of
absorption . It is slower than intramuscular injections but still faster than intradermal injections .
</item 9>

<item 10>
Subcutaneous injection
The injection should be given under the skin , into the fat layer . See also ( edit ) Intramuscular injection Intravenous injection Intradermal injection
References ( edit ) Jump up ^ `` subcutaneous injection '' at Dorland 's Medical Dictionary Jump up ^ `` ISMP 's List of Error - Prone Abbreviations , Symbols ,
and Dose Designations '' ( PDF ) . www.ismp.org. 2013 . Retrieved 13 May 2013 . ^ Jump up to : Taylor , C.R. , Lillis , C. , LeMone , P. , Lynn , P. ( 2011 )
Fundamentals of nursing : The art and science of nursing care . Philadelphia : Lippincott Williams & Wilkins , page 749 . Jump up ^ `` Insulin Administration ''
. Diabetes Care . 25 ( suppl 1 ) : s112 -- s115 . January 2002 . doi : 10.2337 / diacare. 25.2007. s112 . ( hide ) Routes of administration ,
</item 10>
=== END ITEM id=9ede402420143753 ===

=== ITEM id=ae7d56e1e4ae61eb ===
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
`` David Letterman Returns To Television With Netflix Series '' . Deadline . Retrieved January 13 , 2018 . Jump up ^ `` David Letterman makes his return to
television on Netflix '' ( Press release ) . Hollywood , CA : Netflix . August 8 , 2017 . Retrieved August 8 , 2017 . Jump up ^ Hipes , Patrick ( January 5 ,
2018 ) . `` David Letterman 's Netflix Show To Bow January 12 With Barack Obama As First Guest '' . Deadline . Retrieved January 13 , 2018 . Jump up ^ Lewis ,
Hilary ( January 5 , 2018 ) . `` David Letterman 's Netflix Talk Show Sets Barack Obama as First Guest '' . The Hollywood Reporter . Retrieved January 13 , 2018
. Jump up ^ Stuever , Hank ( January 12 , 2018 ) . `` Review : Letterman sits down with Obama , but both men seem rusty and off their game '' . Washington Post
. Retrieved January 13 , 2018 .
</item 1>

<item 2>
My Next Guest Needs No Introduction with David Letterman
My Next Guest Needs No Introduction with David Letterman - wikipedia My Next Guest Needs No Introduction with David Letterman Jump to : navigation , search My
Next Guest Needs No Introduction with David Letterman Genre Talk show Directed by Michael Bonfiglio Presented by David Letterman Composer ( s ) Paul Shaffer
Country of origin United States Original language ( s ) English No. of seasons No. of episodes 4 ( list of episodes ) Production Executive producer ( s ) Justin
Wilkes Dave Sirulnick Jon Kamen Tom Keany Producer ( s ) Mary Barclay Cinematography Declan Quinn Editor ( s ) Jesse Gordon Edward A. Bishop Paul Little Running
time 50 -- 56 minutes Production company ( s ) RadicalMedia Worldwide Pants Inc . Release Original network Netflix Original release January 12 , 2018 ( 2018 -
01 - 12 ) -- present External links Official website My Next Guest Needs No Introduction with David Letterman is an American web television talk show hosted by
David Letterman . The series consists of interviews with one guest per episode both inside and outside a studio setting .
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
`` I Had a Paper Route Too '' Jay - Z April 6 , 2018 ( 2018 - 04 - 06 ) Cold open : None On - location segment : Letterman visits Shangri La Studio , speaks
with music producer Rick Rubin and listens to Madison Ryann Ward recording . 5 `` TBA '' Tina Fey May 4 , 2018 6 `` TBA '' Howard Stern May 31 , 2018 Reception
( edit ) The first season of My Next Guest Needs No Introduction has been met with a positive response from critics . On the review aggregation website Rotten
Tomatoes , the first season holds a 80 % approval rating with an average rating of 6.4 out of 10 based on 25 reviews . Metacritic , which uses a weighted
average , assigned the season a score of 70 out of 100 based on 13 reviews , indicating `` generally favorable reviews '' . References ( edit ) Jump up ^
Andreeva , Nellie ( August 8 , 2017 ) . `` David Letterman Returns To Television With Netflix Series '' . Deadline .
</item 4>

<item 5>
My Next Guest Needs No Introduction with David Letterman
`` You Be the Newsman , I 'll Be Liz Taylor '' George Clooney February 9 , 2018 ( 2018 - 02 - 09 ) Cold open : Letterman and Clooney outside of LAX watching
planes land and eating fast food . On - location segment : At the home of Nick Clooney meeting Iraqi refugee Hazim Avdal whom the Clooney family is sponsoring .
`` You Know , She Has a Nobel Peace Prize '' Malala Yousafzai March 9 , 2018 ( 2018 - 03 - 09 ) Cold open : Letterman accompanies Yousafzai as she leads a tour
group at Lady Margaret Hall , Oxford . On - location segment : Yousafzai continues to lead the tour group as Letterman tags along , the pair browse in the
University of Oxford shop , and Letterman eats lunch with Yousafzai 's father .
</item 5>

<item 6>
My Next Guest Needs No Introduction with David Letterman
On January 5 , 2018 , Netflix released further information about the series . It was announced that Letterman 's guests in the first season would include Barack
Obama , George Clooney , Malala Yousafzai , Jay - Z , Tina Fey , and Howard Stern . It was also announced that the series would premiere on January 12 , 2018
with Letterman 's interview with Obama . This interview was Obama 's first since leaving office . Music ( edit ) The show 's theme song and bridging music were
written and recorded by Letterman 's long - time bandleader , Paul Shaffer . Soon after it was announced publicly that Netflix had given the series an order ,
Shaffer received a phone call from Letterman asking him to work on the show . Soon after , Shaffer began to receive cuts of episodes from the first season and
he started to put music in afterwards where the director thought it was needed . In developing the sound of the show 's music , Shaffer initially looked to
Letterman for guidance .
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
The series consists of interviews with one guest per episode both inside and outside a studio setting . The first season will consist of six episodes and will
include interviews with Barack Obama , George Clooney , Malala Yousafzai , Jay - Z , Tina Fey , and Howard Stern . Obama was the first guest on the show , with
the first episode released on Netflix on January 12 , 2018 . Contents ( hide ) 1 Production 1.1 Development 1.2 Music 2 Episodes 3 Reception 4 References 5
External links Production ( edit ) Development ( edit ) On August 8 , 2017 , it was announced that Netflix had ordered a six - episode interview series starring
David Letterman . Each hour - long episode will feature Letterman conducting a longform conversation with a single guest and in - the - field segments , in
which he will explore topics on his own . On January 5 , 2018 , Netflix released further information about the series .
</item 8>

<item 9>
Graph query row:
show: My Next Guest Needs No Introduction with David Letterman
episode: Premiere of My Next Guest Needs No Introduction with David Letterman
episode_date: January 12, 2018
</item 9>

<item 10>
My Next Guest Needs No Introduction with David Letterman
Washington Post . Retrieved January 13 , 2018 . Jump up ^ Leight , Elias ( February 2 , 2018 ) . `` Paul Shaffer on Writing Letterman 's Netflix Theme Song &
the Value of Humor in Pop '' . Billboard . Retrieved February 9 , 2018 . Jump up ^ Carr , Mary Kate ( April 24 , 2018 ) . `` What 's on Netflix in May : The
best shows and movies to watch this month '' . Entertainment Weekly . Retrieved April 24 , 2018 . Jump up ^ Douglas , Nick ( April 24 , 2018 ) . `` Here 's What
's Coming And Going From Netflix In May 2018 '' . Lifehacker . Retrieved April 24 , 2018 . Jump up ^ `` My Next Guest Needs No Introduction With David Letterman
: Season 1 - Rotten Tomatoes '' . Rotten Tomatoes . Retrieved January 13 , 2018 .
</item 10>
=== END ITEM id=ae7d56e1e4ae61eb ===
