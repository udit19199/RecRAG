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

=== ITEM id=d4c4c71effd93041 ===
Gold sentences and facts:
Sentence 0: The Pakistani National Assembly ( Urdu : قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority .
  Fact 0.0: The Pakistani National Assembly is the lower house of the bicameral Majlis-e-Shura.
  Fact 0.1: The Urdu name of the Pakistani National Assembly is Qaumī Asimbli'e Pākistān.
  Fact 0.2: The Majlis-e-Shura also comprises the President of Pakistan.
  Fact 0.3: The Majlis-e-Shura also comprises the Senate, which is the upper house.
  Fact 0.4: The National Assembly convenes at Parliament House in Islamabad.
  Fact 0.5: The Senate convenes at Parliament House in Islamabad.
  Fact 0.6: The National Assembly is a democratically elected body.
  Fact 0.7: The National Assembly has a total of 332 members.
  Fact 0.8: The members of the National Assembly are referred to as Members of the National Assembly (MNAs).
  Fact 0.9: 272 members of the National Assembly are directly elected.
  Fact 0.10: 70 members of the National Assembly hold reserved seats for women and religious minorities.
  Fact 0.11: A political party must secure 172 seats to obtain a majority.
  Fact 0.12: A political party must secure 172 seats to preserve a majority.

Retrieved items:
<item 1>
National Assembly of Pakistan
The Public Accounts Committee has a special role to review the report of the Auditor General . Senate , the upper house of the Parliament , has equal
representation from the federating units balancing the provincial inequality in the National Assembly , where the number of members is based on population of
the provinces . The Senate 's role is to promote national cohesion and harmony , and work as a stabilizing factor of the federation . The Senate numbers a total
of 104 members who serve six - year terms which are alternated so that half the senators are up for re-election by the electoral college every three years . The
National Assembly consists of 332 members . The Constitution empowers the President to dissolve the National Assembly , but the Senate is not subject to
dissolution . Only the Parliament can amend the Constitution , by two - thirds majority vote separately in each House . Members of National Assembly ( edit )
See also : List of Pakistan National Assembly Seats Qualifications ( edit ) The Constitution of Pakistan lists a number of requirements for members of the
National Assembly in Article 62 .
</item 1>

<item 2>
National Assembly of Pakistan
Seat distribution ( edit ) In the 14th Assembly , the seats have been allocated among the four provinces , federally - administered area , and capital territory
of Pakistan as follows : Province General Women Total Punjab 148 35 183 Sindh 61 14 75 KPK 35 8 43 Balochistan 14 17 FATA 12 - 12 Federal Capital - See also (
edit ) Speaker of National Assembly Senate of Pakistan Politics of Pakistan Prime Minister of Pakistan Constitution of Pakistan President of Pakistan Supreme
Court of Pakistan Pakistan Bar Council Punjab Bar Council Supreme Court Bar Association of Pakistan List of Pakistan National Assembly Seats List of provincial
governments of Pakistan showing provincial assemblies Notes ( edit ) ^ Jump up to : `` PML - N 's Ayaz Sadiq elected Speaker , Murtaza Abbasi Deputy Speaker in
NA '' . Dawn News . 3 June 2013 . Retrieved 6 June 2013 . Jump up ^ `` PPP 's Khursheed Shah declared leader of opposition in NA '' . Dawn News . 7 June 2013 .
Retrieved 7 June 2013 .
</item 2>

<item 3>
National Assembly of Pakistan
National Assembly of Pakistan - wikipedia National Assembly of Pakistan Jump to : navigation , search For List of MNAs elected in Pakistani general election ,
2013 , see List of MNAs elected in Pakistani general election , 2013 . National Assembly of Pakistan قومی اسمبلئ پاکستان ‬ 14th National Assembly of Pakistan
Type Type Lower house of the Parliament of Pakistan Leadership Speaker Ayaz Sadiq ( PML - N ) Since 9 November 2015 Deputy Speaker Javed Abbasi ( PML - N )
Since 3 June 2013 Leader of House Shahid Khaqan Abbasi ( PML - N ) Since 1 August 2017 Leader of Opposition Khurshid Shah ( PPP ) Since 7 June 2013 Structure
Seats 332 ( 3 vacant ) Political groups Government Coalition ( 209 ) PML -- N ( 188 ) JUI -- F ( 13 ) PML -- F ( 5 ) NPP ( 2 ) NP ( 1 ) Opposition ( 130 ) PPP (
47 ) PTI ( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP (
</item 3>

<item 4>
National Assembly of Pakistan
The committees have also been empowered to invite or summon before it any member or any other person having a special interest in relation to any matter under
its consideration and may hear expert evidence and hold public hearing . Composition and elections ( edit ) The composition of the National Assembly is
specified in Article 51 of the Constitution of Pakistan . There are a total of 332 seats in the National Assembly . Of these , 272 are filled by direct
elections . In addition , the Pakistani Constitution reserves 10 seats for religious minorities and 60 seats for women , to be filled by proportional
representation among parties with more than 5 % of the vote . As of 2006 , there are 72 women members in the Assembly . Members of the National Assembly are
elected by the people in competitive multi-party elections , to be held at most five years apart on universal adult franchise . To be a member of electoral
college , according to Article 62 of the Constitution , candidates must be citizens of Pakistan and not less than 18 years of age . 2013 election ( edit ) Main
article : Pakistani general election , 2013 Results of the Pakistani general election ,
</item 4>

<item 5>
National Assembly of Pakistan
631 0.47 % 0 9 National Peoples Party 197,829 0.44 % 0 10 Pakistan Muslim League ( Q ) 1,409,905 3.11 % 0 0 11 Awami National Party 453,057 1.00 % 0 0 12 Qaumi
Watan Party ( Sherpao ) 46,574 0.10 % 0 0 13 Pakistan Muslim League ( Z ) 128,510 0.28 % 0 0 14 National Party 61,148 0.13 % 0 0 15 Awami Muslim League Pakistan
93,046 0.20 % 0 0 16 Awami Jamhuri Ittehad Pakistan 71,175 0.16 % 0 0 17 Balochistan National Party 63,979 0.14 % 0 0 18 All Pakistan Muslim League 54,231 0.12
% 0 0 19 Independents 5,880,658 12.96 % 27 8 0 0 8 Total 45,388,
</item 5>

<item 6>
National Assembly of Pakistan
96 % 27 8 0 0 8 Total 45,388,404 100 % 266 266 60 10 336 Withheld : NA - 103 , NA - 237 ; Postponed : NA - 38 ; Re-poll ordered : NA - 46 ; Terminated : NA - 83
, NA - 254 Source : Detailed Results General Elections - 2013 , Reserve Seats Women , Reserve Seats Minorities Dissolution ( edit ) National Assembly can be
dissolved at the initiative of the Prime Minister . If dissolved , new elections are conducted for the Assembly . Article 58 of the Constitution of Pakistan
deals with the dissolution of the Assembly : 58 . Dissolution of the National Assembly : ( 1 ) The President shall dissolve the National Assembly if so advised
by the Prime Minister ; and the National Assembly shall , unless sooner dissolved , stand dissolved at the expiration of forty - eight hours after the Prime
Minister has so advised .
</item 6>

<item 7>
National Assembly of Pakistan
Area - XII ) NA - 48 ( Islamabad - I ) NA - 49 ( Islamabad - II ) National Assembly constituencies from the Punjab ( NA - 50 to NA - 197 ) NA - 50 ( Rawalpindi
- I ) NA - 51 ( Rawalpindi - II ) NA - 52 ( Rawalpindi - III ) NA - 53 ( Rawalpindi - IV ) NA - 54 ( Rawalpindi - V ) NA - 55 ( Rawalpindi - VI ) NA - 56 (
Rawalpindi - VII ) NA - 57 ( Attock - I ) NA - 58 ( Attock - II ) NA - 59 ( Attock - III ) NA - 60 ( Chakwal - I ) NA - 61 ( Chakwal - II ) NA - 62 ( Jhelum - I
) NA - 63 ( Jhelum - II ) NA - 64 ( Sargodha - I ) NA - 65 ( Sargodha - II ) NA - 66 ( Sargodha - III ) NA - 67 ( Sargodha - IV ) NA
</item 7>

<item 8>
National Assembly of Pakistan
- 26 ( Bannu ) NA - 27 ( Lakki Marwat ) NA - 28 ( Buner ) NA - 29 ( Swat - I ) NA - 30 ( Swat - II ) NA - 31 ( Shangla ) NA - 32 ( Chitral ) NA - 33 ( Upper Dir
- cum - Lower Dir ) NA - 34 ( Lower Dir ) NA - 35 ( Malakand ) National Assembly constituencies from Islamabad and FATA ( NA - 36 to NA - 49 ) NA - 36 ( Tribal
Area - I ) NA - 37 ( Tribal Area - II ) NA - 38 ( Tribal Area - III ) NA - 39 ( Tribal Area - IV ) NA - 40 ( Tribal Area - V ) NA - 41 ( Tribal Area - VI ) NA -
42 ( Tribal Area - VII ) NA - 43 ( Tribal Area - VIII ) NA - 44 ( Tribal Area - IX ) NA - 45 ( Tribal Area - X ) NA - 46 ( Tribal Area - XI ) NA - 47 ( Tribal
Area - XII ) NA - 48 ( Islamabad - I ) NA - 49 ( Islamabad -
</item 8>

<item 9>
National Assembly of Pakistan
( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP ( 3 ) ANP ( 2 ) PML -- Q ( 2 ) AJIP ( 1 ) AML ( 1 ) APML ( 1 ) BNP ( 1 ) PML -- Z ( 1 ) QWP ( 1 ) Ind ( 10 ) Elections Voting
system Mixed member majoritarian ( First past the post for most seats , 60 seats reserved for women and 10 seats reserved for religious minorities by
proportional representation ) Last election 11 May 2013 Meeting place Parliament House , Islamabad Website www.na.gov.pk The Pakistani National Assembly ( Urdu
: قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan
and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad .
</item 9>

<item 10>
National Assembly of Pakistan
The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a
total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for
women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority . Members are elected through the first - past - the
- post system under universal adult suffrage , representing electoral districts known as National Assembly constituencies . According to the constitution , the
70 seats reserved for women and religious minorities are allocated to the political parties according to their proportional representation . Each National
Assembly is formed for a five - year term , commencing from the date of the first sitting , after which it is automatically dissolved . Currently the National
Assembly can not be dissolved by the President of Pakistan , it is dissolved by the Prime Minister of Pakistan . Election for 13th National Assembly was held on
18 February 2008 . On March 17 , 2013 13th National Assembly was dissolved on completion of its five - year term under Article 52 of the Constitution .
</item 10>
=== END ITEM id=d4c4c71effd93041 ===

=== ITEM id=e261d9f0cf7ab86e ===
Gold sentences and facts:
Sentence 0: The earliest predecessors of the eye were photoreceptor proteins that sense light , found even in unicellular organisms , called `` eyespots '' . Eyespots can only sense ambient brightness : they can distinguish light from dark , sufficient for photoperiodism and daily synchronization of circadian rhythms . They are insufficient for vision , as they can not distinguish shapes or determine the direction light is coming from . Eyespots are found in nearly all major animal groups , and are common among unicellular organisms , including euglena . The euglena 's eyespot , called a stigma , is located at its anterior end . It is a small splotch of red pigment which shades a collection of light sensitive crystals . Together with the leading flagellum , the eyespot allows the organism to move in response to light , often toward the light to assist in photosynthesis , and to predict day and night , the primary function of circadian rhythms . Visual pigments are located in the brains of more complex organisms , and are thought to have a role in synchronising spawning with lunar cycles . By detecting the subtle changes in night - time illumination , organisms could synchronise the release of sperm and eggs to maximise the probability of fertilisation .
  Fact 0.0: The earliest predecessors of the eye were photoreceptor proteins that sense light.
  Fact 0.1: The light-sensing photoreceptor proteins that were the earliest predecessors of the eye are found even in unicellular organisms.
  Fact 0.2: These early light-sensing structures are called 'eyespots'.
  Fact 0.3: Eyespots can only sense ambient brightness.
  Fact 0.4: Eyespots can distinguish light from dark.
  Fact 0.5: Sensing ambient brightness is sufficient for photoperiodism.
  Fact 0.6: Sensing ambient brightness is sufficient for daily synchronization of circadian rhythms.
  Fact 0.7: Eyespots are insufficient for vision.
  Fact 0.8: Eyespots cannot distinguish shapes.
  Fact 0.9: Eyespots cannot determine the direction light is coming from.
  Fact 0.10: Eyespots are found in nearly all major animal groups.
  Fact 0.11: Eyespots are common among unicellular organisms.
  Fact 0.12: Euglena is a unicellular organism with an eyespot.
  Fact 0.13: The euglena's eyespot is called a stigma.
  Fact 0.14: The euglena's stigma is located at its anterior end.
  Fact 0.15: The euglena's stigma is a small splotch of red pigment.
  Fact 0.16: The euglena's stigma shades a collection of light-sensitive crystals.
  Fact 0.17: Together with the leading flagellum, the euglena's eyespot allows the organism to move in response to light.
  Fact 0.18: The euglena often moves toward the light to assist in photosynthesis.
  Fact 0.19: The euglena's eyespot allows the organism to predict day and night.
  Fact 0.20: Predicting day and night is the primary function of circadian rhythms.
  Fact 0.21: Visual pigments are located in the brains of more complex organisms.
  Fact 0.22: Visual pigments are thought to have a role in synchronising spawning with lunar cycles.
  Fact 0.23: Organisms could detect subtle changes in night-time illumination.
  Fact 0.24: By detecting night-time illumination changes, organisms could synchronise the release of sperm and eggs.
  Fact 0.25: Synchronising the release of sperm and eggs maximises the probability of fertilisation.

Retrieved items:
<item 1>
Evolution of the eye
This suggests that when the two lineages diverged in the Precambrian , they had only very primitive light receptors , which developed into more complex eyes
independently . Early eyes ( edit ) The basic light - processing unit of eyes is the photoreceptor cell , a specialized cell containing two types of molecules
in a membrane : the opsin , a light - sensitive protein , surrounding the chromophore , a pigment that distinguishes colors . Groups of such cells are termed ``
eyespots '' , and have evolved independently somewhere between 40 and 65 times . These eyespots permit animals to gain only a very basic sense of the direction
and intensity of light , but not enough to discriminate an object from its surroundings . Developing an optical system that can discriminate the direction of
light to within a few degrees is apparently much more difficult , and only six of the thirty - some phyla possess such a system . However , these phyla account
for 96 % of living species . The planarian has `` cup '' eyespots that can slightly distinguish light direction .
</item 1>

<item 2>
Evolution of the eye
149 Jump up ^ Lee , M.S.Y. ; Jago , J.B. ; Garcia - Bellido , D.C. ; Edgecombe , G.E. ; Gehling , J. G ; Paterson , J.R. ( 2011 ) . `` Modern optics in
exceptionally preserved eyes of Early Cambrian arthropods from Australia '' . Nature . 474 : 631 -- 634 . doi : 10.1038 / nature10097 . PMID 21720369 . ^ Jump
up to : Darwin , Charles ( 1859 ) . On the Origin of Species . London : John Murray . Jump up ^ Gehring WJ ( 2005 ) . `` New perspectives on eye development and
the evolution of eyes and photoreceptors '' . J. Hered. 96 ( 3 ) : 171 -- 84 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Jump up ^ Nilsson , D. - E. (
2013 ) . `` Eye evolution and its functional basis '' . Visual Neuroscience.
</item 2>

<item 3>
Evolution of the eye
PMID 25734540 . Greuet , C ( 1968 ) . `` Organisation ultrastructurale de l'ocelle de deux Peridiniens Warnowiidae , Erythropsis pavillardi Kofoid et Swezy et
Warnowia pulchra Schiller '' . Protistologica. 4 : 209 -- 230 . Gregory S. Gavelis , Shiho Hayakawa , Richard A. White III , Takashi Gojobori , Curtis A. Suttle
, Patrick J. Keeling , Brian S. Leander ( 2015 ) . `` Eye - like ocelloids are built from different endosymbiotically acquired components '' . Nature . 523 :
204 -- 7 . Bibcode : 2015Natur. 523 ... 204G . doi : 10.1038 / nature14593 . PMID 26131935 . CS1 maint : Uses authors parameter ( link ) Oakley , Todd H. ;
Speiser , Daniel I. ( 2015 ) . `` How Complexity Originates : The Evolution of Animal Eyes '' .
</item 3>

<item 4>
Graph query row:
entity: eyespots
relationship: RELATED_TO
related: direction and intensity of light
detail: provide a basic sense of
date: None
</item 4>

<item 5>
Evolution of the eye
PMID 7892602 . Jump up ^ Tomarev , S.I. ; Callaerts , P. ; Kos , L. ; Zinovieva , R. ; Halder , G. ; Gehring , W. ; Piatigorsky , J. ( 1997 ) . `` Squid Pax - 6
and eye development '' . Proceedings of the National Academy of Sciences of the United States of America . 94 ( 6 ) : 2421 -- 2426 . Bibcode : 1997PNAS ...
94.2421 T . doi : 10.1073 / pnas. 94.6. 2421 . PMC 20103 . PMID 9122210 . ^ Jump up to : Gehring , W.J. ( 13 January 2005 ) . `` New Perspectives on Eye
Development and the Evolution of Eyes and Photoreceptors '' ( Full text ) . Journal of Heredity . Oxford Journals. 96 ( 3 ) : 171 -- 184 .
</item 5>

<item 6>
Evolution of the eye
1 ( 4 ) : 439 -- 447 . doi : 10.1007 / s12052 - 008 - 0084 - 1 . ISSN 1936 - 6426 . Further reading ( edit ) Lamb TD , Collin SP , Pugh EN ( December 2007 ) .
`` Evolution of the vertebrate eye : opsins , photoreceptors , retina and eye cup '' . Nat. Rev. Neurosci. 8 ( 12 ) : 960 -- 76 . doi : 10.1038 / nrn2283 . PMC
3143066 . PMID 18026166 . Illustration . Review Lamb , TD ( 2011 ) . `` Evolution of the Eye '' ( PDF ) . Scientific American . 305 : 64 -- 69 . doi : 10.1038 /
scientificamerican0711 - 64 . Land , Michael F. ; Nilsson , Dan - Eric ( 2012 ) . `` The origin of vision '' . Animal Eyes ( 2 ed . ) . Oxford : Oxford
University Press .
</item 6>

<item 7>
Evolution of the eye
Oxford Journals. 96 ( 3 ) : 171 -- 184 . doi : 10.1093 / jhered / esi027 . PMID 15653558 . Retrieved 2008 - 04 - 26 . ^ Jump up to : MF Land ; RD Fernald ( 1992
) . `` The Evolution of Eyes '' . Annual Review of Neuroscience. 15 : 1 -- 29 . doi : 10.1146 / annurev. ne. 15.030192. 000245 . PMID 1575438 . Jump up ^ Autrum
, H ( 1979 ) . `` Introduction '' . In H. Autrum . Comparative Physiology and Evolution of Vision in Invertebrates - A : Invertebrate Photoreceptors . Handbook
of Sensory Physiology . VII / 6A . New York : Springer - Verlag . pp. 6 -- 9 . ISBN 3 - 540 - 08837 - 7 .
</item 7>

<item 8>
Graph query row:
entity: eyespots
relationship: PART_OF
related: photoreceptor cell
detail: groups of photoreceptor cells are termed eyespots
date: None
</item 8>

<item 9>
Evolution of the eye
Evolution of the eye - wikipedia Evolution of the eye Jump to : navigation , search Major stages in the evolution of the eye in vertebrates . The evolution of
the eye is attractive to study , because the eye distinctively exemplifies an analogous organ found in many animal forms . Complex , image - forming eyes have
evolved independently several times . The evolutionary geneticist Walter Gering holds a different opinion . According his theory , the genes engaged with
photoreceptor cells ( so called Pax 6 genes ) appeared probably at once in the life history , at a very early stage of the evolution . Probably , the complex ,
image forming kinds of eyes are at least five : Chidaria - lens eye , complex eye ; Molluscs - lens eye , everse retina ; Annelids - lens eye , everse retina ;
Arthropods - facet eye ; Vertebrate - lens eye , inverse retina Complex eyes appeared first within the few million years of the Cambrian explosion . From before
the Cambrian , no evidence of eyes has survived , but diverse eyes are known from the Burgess shale of the Middle Cambrian , and from the slightly older Emu Bay
Shale .
</item 9>

<item 10>
Evolution of the eye
`` Eye evolution and its functional basis '' . Visual Neuroscience. 30 : 5 -- 20 . doi : 10.1017 / s0952523813000035 . ^ Jump up to : Nilsson , D. - E. ; Pelger
, S. ( 1994 ) . `` A pessimistic estimate of the time required for an eye to evolve '' . Proceedings of the Royal Society B : Biological Sciences . 256 : 53 --
58 . doi : 10.1098 / rspb. 1994.0048 . PMID 8008757 . Jump up ^ Young , G.C. ( 2008 ) . `` Early evolution of the vertebrate eye -- fossil evidence '' . Evo Edu
Outreach. 1 : 427 -- 438 . doi : 10.1007 / s12052 - 008 - 0087 - y . Jump up ^ Parker , A.R. ( 2009 ) . `` On the origin of optics '' . Optics & Laser
Technology . 43 ( 2 ) : 323 -- 329 .
</item 10>
=== END ITEM id=e261d9f0cf7ab86e ===

=== ITEM id=0dff5d2be371f233 ===
Gold sentences and facts:
Sentence 0: Simplified reaction : C H O ( s ) + 6 O ( g ) → 6 CO ( g ) + 6 H O ( l ) + heat ΔG = − 2880 kJ per mol of C H O
  Fact 0.0: The simplified reaction is C H O (s) + 6 O (g) → 6 CO (g) + 6 H O (l) + heat.
  Fact 0.1: The free energy change ΔG is −2880 kJ per mol of C H O.

Retrieved items:
<item 1>
Cellular respiration
Mitochondrial outer membrane is omitted . According to some of newer sources the ATP yield during aerobic respiration is not 36 -- 38 , but only about 30 -- 32
ATP molecules / 1 molecule of glucose , because : ATP : NADH + H and ATP : FADH ratios during the oxidative phosphorylation appear to be not 3 and 2 , but 2.5
and 1.5 respectively . Unlike in the substrate - level phosphorylation , the stoichiometry here is difficult to establish . ATP synthase produces 1 ATP / 3 H.
However the exchange of matrix ATP for cytosolic ADP and Pi ( antiport with OH or symport with H ) mediated by ATP -- ADP translocase and phosphate carrier
consumes 1 H / 1 ATP as a result of regeneration of the transmembrane potential changed during this transfer , so the net ratio is 1 ATP : 4 H . The
mitochondrial electron transport chain proton pump transfers across the inner membrane 10 H / 1 NADH + H ( 4 + 2 + 4 ) or 6 H / 1 FADH ( 2 + 4 ) .
</item 1>

<item 2>
Cellular respiration
Obviously this reduces the theoretical efficiency of the whole process and the likely maximum is closer to 28 -- 30 ATP molecules . In practice the efficiency
may be even lower because the inner membrane of the mitochondria is slightly leaky to protons . Other factors may also dissipate the proton gradient creating an
apparently leaky mitochondria . An uncoupling protein known as thermogenin is expressed in some cell types and is a channel that can transport protons . When
this protein is active in the inner membrane it short circuits the coupling between the electron transport chain and ATP synthesis . The potential energy from
the proton gradient is not used to make ATP but generates heat . This is particularly important in brown fat thermogenesis of newborn and hibernating mammals .
Stoichiometry of aerobic respiration and most known fermentation types in eucaryotic cell . Numbers in circles indicate counts of carbon atoms in molecules , C6
is glucose C H O , C1 carbon dioxide CO . Mitochondrial outer membrane is omitted .
</item 2>

<item 3>
Cellular respiration
Glycolysis pay - off phase Substrate - level phosphorylation 2 NADH 3 or 5 Oxidative phosphorylation : Each NADH produces net 1.5 ATP ( instead of usual 2.5 )
due to NADH transport over the mitochondrial membrane Oxidative decarboxylation of pyruvate 2 NADH 5 Oxidative phosphorylation Krebs cycle Substrate - level
phosphorylation 6 NADH 15 Oxidative phosphorylation 2 FADH Oxidative phosphorylation Total yield 30 or 32 ATP From the complete oxidation of one glucose
molecule to carbon dioxide and oxidation of all the reduced coenzymes . Although there is a theoretical yield of 38 ATP molecules per glucose during cellular
respiration , such conditions are generally not realized because of losses such as the cost of moving pyruvate ( from glycolysis ) , phosphate , and ADP (
substrates for ATP synthesis ) into the mitochondria . All are actively transported using carriers that utilize the stored energy in the proton electrochemical
gradient .
</item 3>

<item 4>
Cellular respiration
However , if oxygen is not present , fermentation of the pyruvate molecule will occur . In the presence of oxygen , when acetyl - CoA is produced , the molecule
then enters the citric acid cycle ( Krebs cycle ) inside the mitochondrial matrix , and is oxidized to CO while at the same time reducing NAD to NADH . NADH can
be used by the electron transport chain to create further ATP as part of oxidative phosphorylation . To fully oxidize the equivalent of one glucose molecule ,
two acetyl - CoA must be metabolized by the Krebs cycle . Two waste products , H O and CO , are created during this cycle . The citric acid cycle is an 8 - step
process involving 18 different enzymes and co-enzymes . During the cycle , acetyl - CoA ( 2 carbons ) + oxaloacetate ( 4 carbons ) yields citrate ( 6 carbons )
, which is rearranged to a more reactive form called isocitrate ( 6 carbons ) .
</item 4>

<item 5>
Cellular respiration
5 ATP ( if glycerol phosphate shuttle transfers hydrogen atoms ) or 2 × 2.5 ATP ( malate - aspartate shuttle ) 2 NADH + H from the oxidative decarboxylation of
pyruvate and 6 from Krebs cycle : 8 × 2.5 ATP 2 FADH from the Krebs cycle : 2 × 1.5 ATP Altogether this gives 4 + 3 ( or 5 ) + 20 + 3 = 30 ( or 32 ) ATP per
molecule of glucose The total ATP yield in ethanol or lactic acid fermentation is only 2 molecules coming from glycolysis , because pyruvate is not transferred
to the mitochondrion and finally oxidized to the carbon dioxide ( CO ) , but reduced to ethanol or lactic acid in the cytoplasm . Fermentation Main article :
Fermentation Without oxygen , pyruvate ( pyruvic acid ) is not metabolized by cellular respiration but undergoes a process of fermentation .
</item 5>

<item 6>
Cellular respiration
Contents ( hide ) 1 Aerobic respiration 1.1 Aerobic respiration summed up 1.2 Glycolysis 1.3 Oxidative decarboxylation of pyruvate 1.4 Citric acid cycle 1.5
Oxidative phosphorylation 2 Efficiency of ATP production 3 Fermentation 4 Anaerobic respiration 5 See also 6 References 7 External links Aerobic respiration
Aerobic respiration ( red arrows ) is the main means by which both fungi and animals utilize chemical energy in the form of organic compounds that were
previously created through photosynthesis ( green arrow ) . Aerobic respiration requires oxygen ( O ) in order to create ATP . Although carbohydrates , fats ,
and proteins are consumed as reactants , it is the preferred method of pyruvate breakdown in glycolysis and requires that pyruvate enter the mitochondria in
order to be fully oxidized by the Krebs cycle .
</item 6>

<item 7>
Cellular respiration
Aerobic respiration summed up 1 . Glycolysis : -- - 2 ATPs + Glucose → 2 Pyruvic Acid + 4 Hydrogen + 4 ATPs 2 . Formation of Acetyl CoA : -- - 2 Pyruvic Acid +
2 CoA → 2 Acetyl CoA + 2 Carbon Dioxide + 2 Hydrogen 3 . Krebs Cycle : -- - 2 Acetyl CoA + 3 O2 → 6 Hydrogen + 4 Carbon Dioxide + 2 ATPs 4 . Electron Transport
System : -- - 12 Hydrogen + 3 O2 → 6 Water + 32 ATPs Overall Reaction : -- - Glucose + 6 O2 → 6 Carbon Dioxide + 6 Water + 36 ATPs Glycolysis Out of the
cytoplasm it goes into the Krebs cycle with the acetyl CoA . It then mixes with CO and makes 2 ATP , NADH , and FADH . From there the NADH and FADH go into the
NADH reductase , which produces the enzyme .
</item 7>

<item 8>
Cellular respiration
The products of this process are carbon dioxide and water , but the energy transferred is used to break bonds in ADP as the third phosphate group is added to
form ATP ( adenosine triphosphate ) , by substrate - level phosphorylation , NADH and FADH Simplified reaction : C H O ( s ) + 6 O ( g ) → 6 CO ( g ) + 6 H O (
l ) + heat ΔG = − 2880 kJ per mol of C H O The negative ΔG indicates that the reaction can occur spontaneously . The potential of NADH and FADH is converted to
more ATP through an electron transport chain with oxygen as the `` terminal electron acceptor '' . Most of the ATP produced by aerobic cellular respiration is
made by oxidative phosphorylation . This works by the energy released in the consumption of pyruvate being used to create a chemiosmotic potential by pumping
protons across a membrane . This potential is then used to drive ATP synthase and produce ATP from ADP and a phosphate group .
</item 8>

<item 9>
Cellular respiration
The electrons are finally transferred to exogenous oxygen and , with the addition of two protons , water is formed . Efficiency of ATP production The table
below describes the reactions involved when one glucose molecule is fully oxidized into carbon dioxide . It is assumed that all the reduced coenzymes are
oxidized by the electron transport chain and used for oxidative phosphorylation . Step coenzyme yield ATP yield Source of ATP Glycolysis preparatory phase − 2
Phosphorylation of glucose and fructose 6 - phosphate uses two ATP from the cytoplasm .
</item 9>

<item 10>
Cellular respiration
Biology textbooks often state that 38 ATP molecules can be made per oxidised glucose molecule during cellular respiration ( 2 from glycolysis , 2 from the Krebs
cycle , and about 34 from the electron transport system ) . However , this maximum yield is never quite reached because of losses due to leaky membranes as well
as the cost of moving pyruvate and ADP into the mitochondrial matrix , and current estimates range around 29 to 30 ATP per glucose . Aerobic metabolism is up to
15 times more efficient than anaerobic metabolism ( which yields 2 molecules ATP per 1 molecule glucose ) . However some anaerobic organisms , such as
methanogens are able to continue with anaerobic respiration , yielding more ATP by using other inorganic molecules ( not oxygen ) as final electron acceptors in
the electron transport chain . They share the initial pathway of glycolysis but aerobic metabolism continues with the Krebs cycle and oxidative phosphorylation
. The post-glycolytic reactions take place in the mitochondria in eukaryotic cells , and in the cytoplasm in prokaryotic cells . Aerobic respiration summed up 1
.
</item 10>
=== END ITEM id=0dff5d2be371f233 ===

=== ITEM id=a256be2001e725d2 ===
Gold sentences and facts:
Sentence 0: The National Legal Aid & Defender Association ( NLADA ) is the oldest and largest national , nonprofit membership organization devoted to advocating equal justice for all Americans and was established in 1911 .
  Fact 0.0: The National Legal Aid & Defender Association (NLADA) is the oldest national membership organization.
  Fact 0.1: The National Legal Aid & Defender Association (NLADA) is the largest national membership organization.
  Fact 0.2: The National Legal Aid & Defender Association (NLADA) is nonprofit.
  Fact 0.3: The National Legal Aid & Defender Association (NLADA) is devoted to advocating equal justice for all Americans.
  Fact 0.4: The National Legal Aid & Defender Association (NLADA) was established in 1911.

Retrieved items:
<item 1>
American Bar Association Model Rules of Professional Conduct
Jump up ^ http://www.mebaroverseers.org/attorney_regulation/maine_conduct_rules.html Jump up ^
http://www.maine.gov/tools/whatsnew/index.php?topic=mebar_overseers_bar_rules&id=88185&v=article Jump up ^ http://www.cnmilaw.org/pdf/court_rules/R07.pdf
Northern Mariana Islands Rules of Attorney Discipline and Procedure . ^ Jump up to : Rogers , Joan C. ( 2 October 2014 ) . `` California Justices Tell State Bar
to Redo Proposals for Updating Lawyer Conduct Rules '' . ABA / BNA Lawyers ' Manual on Professional Conduct . Bloomberg BNA . Retrieved 15 October 2014 . ^ Jump
up to : Zitrin , Richard ( 10 October 2014 ) . `` Viewpoint : State Supreme Court Resets Ethics Rewrite '' . The Recorder . ALM Media Properties , LLC .
Retrieved 15 October 2014 . Jump up ^ http://www.calbar.ca.gov/AboutUs/BoardofTrustees.aspx Board of Trustees , State Bar of California .
</item 1>

<item 2>
American Bar Association Model Rules of Professional Conduct
In contrast to most Commonwealth countries , where transactional lawyers and litigators are separately licensed as `` solicitors '' and `` barristers , ''
respectively ; all American attorneys are licensed to practice as both solicitors and barristers ( therefore these terms are almost never used ) . Likewise ,
whereas solicitors in England and Wales are professionally self - regulated by the private - sector Law Society of England and Wales while barristers are self -
regulated by the private - sector General Council of the Bar , attorneys in all states except California , all territories and the District of Columbia are
generally regulated by the public sector judicial branch of state , district or territorial government , as officers of the court . The legal profession in
California is regulated jointly by the California Supreme Court , mostly through its agency the State Bar of California and by the California Legislature ,
which exercises legislative control over all professions . Within the public sector regulatory agencies , boards and organisations , officers are invariably
licensed members of the pertinent profession while key managerial and general staff typically are not . The Executive Director and Deputy Executive Director of
the State Bar of California , for example , are active members of the State Bar .
</item 2>

<item 3>
American Bar Association Model Rules of Professional Conduct
California ( edit ) California still has its own set of unique rules governing professional attorney conduct . Although many of the rules are similar and
sometimes identical to the ABA Model Rules , the California rules differ markedly from the ABA models in both format , scope and content . For example , under
the ABA Model Rules , an attorney may be disciplined for the appearance of impropriety , regardless whether any actual impropriety occurred ; whereas the
California rules permit discipline for only actual impropriety . This reflects the nature of California law . It is unlikely that California will ever adopt the
ABA Model Rules apart beyond harmonising the format and existing California rules that are identical or nearly identical to ABA Model Rules . Reasons for this
reluctance include the fact there are more than 280,000 attorneys licensed in California , so a drastic change from the existing rules to completely different
rules would shock the profession and , ultimately , would most likely fail . Additionally , many of California 's rules of professional conduct are established
by statute and , thus , would require action by the California Legislature to change them ; or , alternatively , action by the people of California through the
voter initiative process .
</item 3>

<item 4>
American Bar Association Model Rules of Professional Conduct
External links ( edit ) The ABA Model Rules of Professional Conduct Pre-2002 Changes to the Model Rules since its adoption in 1983 from Cornell Law School
Retrieved from `` https://en.wikipedia.org/w/index.php?title=American_Bar_Association_Model_Rules_of_Professional_Conduct&oldid=789593953 '' Categories : 1983
in law Legal ethics Codes of conduct American Bar Association United States law 1983 documents Hidden categories : Articles containing potentially dated
statements from 2015 All articles containing potentially dated statements Talk Contents About Wikipedia 한국어 Suomi Edit links This page was last edited on 8 July
2017 , at 08 : 30 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree
to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
American Bar Association Model Rules of Professional Conduct
American Bar Association Model Rules of professional conduct - wikipedia American Bar Association Model Rules of professional conduct Jump to : navigation ,
search Professional responsibility Duties to the client Confidentiality Avoiding conflict of interest Due diligence and Competence ( law ) Avoid commingling
Avoid self - dealing Effective assistance Avoid fee splitting Withdrawal from representation Duties to the court Disclosure of perjury Disclosure of adverse
authority Duties to the profession Limitations on legal advertising Report misconduct Sources of law ABA Model Rules Penalties for misconduct Disbarment
Judicial misconduct The ABA Model Rules of Professional Conduct , created by the American Bar Association ( ABA ) , are a set of rules that prescribe baseline
standards of legal ethics and professional responsibility for lawyers in the United States . They were promulgated by the ABA House of Delegates upon the
recommendation of the Kutak Commission in 1983 . The rules are merely recommendations , or models , ( hence the name `` Model Rules '' ) and are not themselves
binding . However , having a common set of Model Rules facilitates a common discourse on legal ethics , and simplifies professional responsibility training as
well as the day - to - day application of such rules .
</item 5>

<item 6>
American Bar Association Model Rules of Professional Conduct
While either of their secretaries could be a member of the State Bar , they most likely are not . Similarly , the Board of Trustees that oversee the business
operations of the State Bar of California under supervision of the California Supreme Court is legally required to be a mix of attorneys licensed to practice in
California and `` public members '' who may not be licensed attorneys . So , while American attorneys are regulated by public sector agencies instead of private
sector law societies , most of the people who develop the regulatory policies are practising attorneys and sitting judges , so there is a degree of self -
regulation but not nearly to the same degree that solicitors and barristers in England and Wales , and other Commonwealth jurisdictions are self - regulated by
private sector associations . See also ( edit ) American Bar Association Model Code of Professional Responsibility American Bar Association References ( edit )
Jump up ^ Press Release : New Attorney Rules of Professional Conduct Announced , Communications Office of the New York Courts , 17 December 2008 .
</item 6>

<item 7>
American Bar Association Model Rules of Professional Conduct
Even though New York did not adopt the Model Rules verbatim , the advantage of adopting its overall structure is that it simplifies the professional
responsibility training of New York lawyers , and makes it easier for out - of - state lawyers to conform their conduct to New York rules by simply comparing
their home state 's version of the Model Rules to New York 's version . In June 2009 , the Supreme Court of Maine approved the adoption of the Model Rules in
that state , effective August 1 , 2009 . Among the portions of the Model Rules that the Maine Rules of Professional Conduct do not include is Model Rule 1.8 ( j
) ( 2002 ) categorically prohibiting sexual relations between lawyer and client . In March 2015 , the Supreme Court of the Commonwealth of the Northern Mariana
Islands adopted its current Rules of Attorney Discipline and Procedure , which incorporates the ABA Model Rules by reference and expands upon them by adding
another 22 local rules of professional conduct . California ( edit ) California still has its own set of unique rules governing professional attorney conduct .
</item 7>

<item 8>
National Legal Aid & Defender Association
Other municipalities followed suit , and in the first decades of the 20th century most major cities had opened legal aid societies . In 1911 , legal aid
societies joined together to form the National Alliance of Legal Aid Societies . Arthur von Briesen of the Legal Aid Society of New York was the first president
of the organization that became the National Legal Aid & Defender Association ( NLADA ) in 1949 . The concept of free legal assistance for the poor was promoted
by the publication of Reginald Heber Smith 's Justice and the Poor in 1919 . Smith challenged the legal profession to consider it an obligation to see that
access to justice was available to all , without regard to ability to pay . `` Without equal access to the law , '' he wrote , `` the system not only robs the
poor of their only protection , but places in the hands of their oppressors the most powerful and ruthless weapon ever invented . As a result of Smith 's book ,
the American Bar Association created the Special Committee on Legal Aid Work . By the middle of the 20th century , virtually every major metropolitan area had
some kind of legal aid program .
</item 8>

<item 9>
American Bar Association Model Rules of Professional Conduct
As of 2015 , 49 states and four territories have adopted the rules in whole or in part , of which the most recent to do so was the Commonwealth of the Northern
Mariana Islands in March 2015 . California is the only state that has not adopted the ABA Model Rules , while Puerto Rico is the only U.S. jurisdiction outside
of confederation has not adopted them but instead has its own Código de Ética Profesional . Contents ( hide ) 1 State , commonwealth and territorial adoptions
of the Rules 1.1 California 2 Professional regulation 3 See also 4 References 5 External links State , Commonwealth and territorial adoptions of the Rules (
edit ) The American Bar Association is a private sector voluntary professional association . As such , it has no lawmaking power or regulatory authority , but
it is just like any other nongovernmental bar association . Accordingly , the Model Rules are not legally binding on ABA members or within any particular
jurisdiction .
</item 9>

<item 10>
American Bar Association Model Rules of Professional Conduct
Another important factor is that the legal profession is regulated very differently in California than it is in the rest of the United States . Outside of
California , the profession is regulated solely by the judicial branch of state government . Within California , the State Bar of California operates as a state
government agency under the supervision of the California Supreme Court ; but the State Bar was created by the California Legislature , and the Legislature has
full power to regulate the legal profession , if it so wishes . In particular , the State Bar can not collect annual member dues unless and until the California
Legislature passes an annual State Bar Dues Bill , which the governor must sign into law . When Governor Pete Wilson vetoed the dues bill in 1997 , the State
Bar effectively shut down operations except for self - funded activities like the California Bar Exam . In 2010 , Governor Arnold Schwarzenegger vetoed the dues
bill but a crisis was averted on that occasion . But the most important reason the California Bar is reluctant to adopt the ABA Model Rules is that most members
believe the ABA Model Rules would not sufficiently protect the California public from harm .
</item 10>
=== END ITEM id=a256be2001e725d2 ===

=== ITEM id=b0e27e4f9366ba99 ===
Gold sentences and facts:
Sentence 0: A. planci has a very wide Indo - Pacific distribution . It is perhaps most common in Australia , but can occur at tropical and subtropical latitudes from the Red Sea and the east African coast across the Indian Ocean , and across the Pacific Ocean to the west coast of Central America . It occurs where coral reefs or hard coral communities occur in this region .
  Fact 0.0: A. planci has a very wide Indo-Pacific distribution.
  Fact 0.1: A. planci is perhaps most common in Australia.
  Fact 0.2: A. planci can occur at tropical and subtropical latitudes from the Red Sea.
  Fact 0.3: A. planci can occur at tropical and subtropical latitudes from the east African coast.
  Fact 0.4: A. planci can occur across the Indian Ocean.
  Fact 0.5: A. planci can occur across the Pacific Ocean to the west coast of Central America.
  Fact 0.6: A. planci occurs where coral reefs or hard coral communities occur in this region.

Retrieved items:
<item 1>
Crown-of-thorns starfish
An appraisal of the evidence '' . Marine Pollution Bulletin . 51 ( 1 -- 4 ) : 266 -- 78 . doi : 10.1016 / j. marpolbul. 2004.10. 035 . PMID 15757727 . Jump up ^
Fabricius , K.E. ; Okaji , K. ; De'Ath , G. ( 2010 ) . `` Three lines of evidence to link outbreaks of the crown - of - thorns seastar Acanthaster planci to the
release of larval food limitation '' . Coral Reefs. 29 ( 3 ) : 593 -- 605 . Bibcode : 2010CorRe ... 29 ... 593F . doi : 10.1007 / s00338 - 010 - 0628 - z . Jump
up ^ Kaufman , MG ; Goodfriend , W ; Kohler - Garrigan , A ; Walker , ED ; Klug , MJ ( 2002 ) .
</item 1>

<item 2>
Crown-of-thorns starfish
on Indo - Pacific Marine Biology , Agana ( Guam ) . pp. Abstract . Jump up ^ Babcock ; Mundy and Whitehead ( 1994 ) . `` Sperm diffusion models and in situ
confirmation of long - distance fertilization in the free - spawning asteroid Acanthaster planci '' . Biological Bulletin . 186 : 17 -- 28 . doi : 10.2307 /
1542033 . JSTOR 1542033 . Jump up ^ Beach , DH ; Hanscomb , NJ ; Ormond , RFG. ( 1975 ) . `` Spawning pheromone in crown - of - thorns starfish '' . Nature .
254 ( 5496 ) : 135 -- 136 . Bibcode : 1975Natur. 254 ... 135B . doi : 10.1038 / 254135a0 . PMID 1117997 . Jump up ^ Sweatman , HPA. ( 1995 ) . `` A field study
of fish predation on juvenile crown - of - thorns starfish '' .
</item 2>

<item 3>
Crown-of-thorns starfish
A. planci has a very wide Indo - Pacific distribution . It is perhaps most common in Australia , but can occur at tropical and subtropical latitudes from the
Red Sea and the east African coast across the Indian Ocean , and across the Pacific Ocean to the west coast of Central America . It occurs where coral reefs or
hard coral communities occur in this region .
</item 3>

<item 4>
Crown-of-thorns starfish
1365 - 2486.2006. 01252. x . Jump up ^ Wilson , SK ; Dolman , AM ; Cheal , AJ ; Emslie , MJ Pratchett ; et al. ( 2009 ) . `` Maintenance of fish diversity on
disturbed coral reefs '' . Coral reefs. 28 ( 1 ) : 3 -- 14 . Bibcode : 2009CorRe ... 28 ... 3W . doi : 10.1007 / s00338 - 008 - 0431 - 2 . Jump up ^ Birkeland
and Lucas ( 1990 ) . Acanthaster planci : Major Management Problem of Coral Reefs . CRC Press . pp. 36 -- 40 . ISBN 0 - 8493 - 6599 - 6 . ^ Jump up to : Moran ,
PJ ( 1986 ) . `` The Acanthaster phenomenon '' . Oceanogr . Mar. Biol. Annu. Rev. 24 : 379 -- 480 .
</item 4>

<item 5>
Crown-of-thorns starfish
Its prehensile ability arises from the two rows of numerous tube feet that extend to the tip of each arm . In being multiple - armed , it has lost the five-fold
symmetry ( pentamerism ) typical of starfish , although it begins its life cycle with this symmetry . Adult crown - of - thorns starfish normally range in size
from 25 to 35 cm ( 9.8 to 13.8 in ) . They have up to 21 arms . Although the body of the crown of thorns has a stiff appearance , it is able to bend and twist
to fit around the contours of the corals on which it feeds . The underside of each arm has a series of closely fitting plates which form a groove and extend in
rows to the mouth . They are usually of subdued colours , pale brown to grey - green , but they may be garish with bright warning colours in some parts of their
wide range . The long , sharp spines on the sides of the starfish 's arms and upper ( aboral ) surface resemble thorns and create a crown - like shape , giving
the creature its name .
</item 5>

<item 6>
Crown-of-thorns starfish
ISBN 978 - 0 - 412 - 03541 - 8 . Jump up ^ Caso , M.J. ( 1974 ) . `` External morphology of Acanthaster planci ( Linnaeus ) '' . Journal of the Marine
Biological Associataion of India . 16 ( 1 ) : 83 -- 93 . Jump up ^ `` Crown of Thorns Starfish ( Acanthaster Planci ) . '' Crown of Thorns Starfish Videos ,
Photos and Facts . Web . 5 November 2014 . < http://www.arkive.org/crown-of-thorns-starfish/acanthaster-planci/ > . Jump up ^ Shedd , John G. ( 2006 ) . ``
Crown of Thorns Sea Star '' . Shedd Aquarium . Archived from the original on 2014 - 02 - 22 . Retrieved 2013 - 06 - 29 . ^ Jump up to : Lucas , J.S. ; Jones ,
M.M. ( 1976 ) .
</item 6>

<item 7>
Crown-of-thorns starfish
00775630.167. 1 . Jump up ^ Kenchington , R ( 1978 ) . `` The crown - of - thorns crisis in Australia : A retrospective analysis '' . Environmental
Conservation. 5 ( 1 ) : 11 -- 20 . doi : 10.1017 / s0376892900005191 . Jump up ^ James , Peter ( 1976 ) . Requiem for the Reef . Brisbane : Foundation Press .
ISBN 0 - 9597383 - 0 - 4 . Jump up ^ Brown , and Willey ( 1972 ) . Crown of Thorns : The Death of the Great Barrier Reef ? . Angus and Robertson . ISBN 0 - 207
- 12390 - X . Jump up ^ Reichelt , RE ; Bradbury , RH ; Moran , P J. ( 1990 ) . `` The crown - of - thorns starfish , Acanthaster planci , on the Great Barrier
Reef '' . Mathematical and Computer Modelling.
</item 7>

<item 8>
Crown-of-thorns starfish
This article 's use of external links may not follow Wikipedia 's policies or guidelines . Please improve this article by removing excessive or inappropriate
external links , and converting useful links where appropriate into footnote references . ( March 2016 ) ( Learn how and when to remove this template message )
An overview of the crown - of - thorns starfish as observed on the Great Barrier Reef Current status of crown - of - thorns starfish on Australia 's Great
Barrier Reef History of outbreaks of crown - of - thorns starfish on Australia 's Great Barrier Reef since 1986 ( Animation ) Recent information on crown - of -
thorns starfish on Australia 's Great Barrier Reef Controlling crown - of - thorns starfish Crown - of - thorn outbreak simulation Microdocs : Crown - of -
thorns Acanthaster planci expedition 1969 to Eastern Caroline Islands An infographic relating to the crown - of - thorns starfish in the Great Barrier Reef
Crown of Thorns starfish 1971 Fathom magazine ( Australia ) Taxon identifiers Wd : Q26634 ADW : Acanthaster_planci ARKive : acanthaster - planci BioLib :
</item 8>

<item 9>
Crown-of-thorns starfish
It occurs where coral reefs or hard coral communities occur in this region . Contents ( hide ) 1 Description 2 Taxonomy 2.1 Family 2.2 Genus and species 2.3
Genetic studies 3 Biology 3.1 Toxins 3.2 Behaviour 3.3 Predators 4 Life - cycle 4.1 Gametes and embryos 4.2 Larval stages 4.3 Metamorphosis , development and
growth 4.4 Field observations of life - cycle 5 Ecology 5.1 Ecological impact on reefs 5.2 Population outbreaks 5.3 Causes of population outbreaks 6 Population
control 7 References 8 External links Description ( edit ) Unusual brightly coloured crown - of - thorns starfish , Thailand The body form of the crown - of -
thorns starfish is fundamentally the same as that of a typical starfish , with a central disk and radiating arms . Its special traits , however , include being
disc - shaped , multiple - armed , flexible , prehensile , and heavily spined , and having a large ratio of stomach surface to body mass .
</item 9>

<item 10>
Crown-of-thorns starfish
The starfish show preferences between the hard corals on which they feed . They tend to feed on branching corals and table - like corals , such as Acropora
species , rather than on more rounded corals with less exposed surface area , such as Porites species . Avoidance of Porites and some other corals may also be
due to resident bivalve molluscs and polychaete worms in the surface of the coral which discourage the starfish . Similarly , some symbionts , such as small
crabs , living within the complex structures of branching corals , may ward off the starfish as it seeks to spread its stomach over the coral surface . In reef
areas of low densities of hard coral , reflecting the nature of the reef community or due to feeding by high density crown - of - thorns , the starfish may be
found feeding on soft corals ( Alcyonacea ) . The starfish are cryptic in behavior during their first two years , emerging at night to feed . They usually
remain so as adults when solitary . The only evidence of a hidden individual may be white feeding scars on adjacent coral .
</item 10>
=== END ITEM id=b0e27e4f9366ba99 ===

=== ITEM id=c1fe71ecfa790e26 ===
Gold sentences and facts:
Sentence 0: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting , a feat matched only by fellow Australian Heath Ledger 32 years later ; Finch had suffered a fatal heart attack in mid-January . Beatrice Straight set another record by becoming the actor with the shortest performance ever in a film to win an acting Oscar , with only five minutes and two seconds of screen - time in Network . Network , along with All the President 's Men , were the two biggest champs of the ceremony with four Oscars each , but Best Picture and Best Director ultimately went to Rocky .
  Fact 0.0: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting.
  Fact 0.1: Peter Finch's posthumous acting Oscar was matched only by fellow Australian Heath Ledger, 32 years later.
  Fact 0.2: Peter Finch had suffered a fatal heart attack in mid-January.
  Fact 0.3: Beatrice Straight set a record as the actor with the shortest performance ever in a film to win an acting Oscar.
  Fact 0.4: Beatrice Straight's performance in Network had only five minutes and two seconds of screen time.
  Fact 0.5: Beatrice Straight won an acting Oscar for her performance in Network.
  Fact 0.6: Network and All the President's Men were the two biggest winners of the ceremony.
  Fact 0.7: Network won four Oscars.
  Fact 0.8: All the President's Men won four Oscars.
  Fact 0.9: Best Picture and Best Director ultimately went to Rocky.

Retrieved items:
<item 1>
49th Academy Awards
producers John G. Avildsen -- Rocky Alan J. Pakula -- All the President 's Men Ingmar Bergman -- Face to Face Sidney Lumet -- Network Lina Wertmüller -- Seven
Beauties Best Actor Best Actress Peter Finch -- Network as Howard Beale Robert De Niro -- Taxi Driver as Travis Bickle Giancarlo Giannini -- Seven Beauties as
Pasqualino William Holden -- Network as Max Schumacher Sylvester Stallone -- Rocky as Rocky Balboa Faye Dunaway -- Network as Diana Christensen Marie -
Christine Barrault -- Cousin Cousine as Marthe Talia Shire -- Rocky as Adriana Pennino Sissy Spacek -- Carrie as Carrie White Liv Ullmann -- Face to Face as
Jenny Isaksson Best Supporting Actor Best Supporting Actress Jason Robards -- All the President 's Men as Ben Bradlee Ned Beatty -- Network as Arthur Jensen
Burgess Meredith -- Rocky as `` Mickey '' Goldmill Laurence Olivier -- Marathon Man as Dr.
</item 1>

<item 2>
Summer of '42
On this point , Raucher said in May 2002 that his ten percent of the gross , in addition to royalties from book sales , `` has paid bills ever since . '' As
well as being a commercial success , Summer of ' 42 also received rave critical reviews . It went on to be nominated for over a dozen awards , including Golden
Globe Awards for `` Best Motion Picture -- Drama '' and `` Best Director '' , and five Academy Award nominations for Best Original Music Score , Best
Cinematography , Best Editing , Best Writing - Story and Best Screenplay . Ultimately , the film won two awards : the 1972 44th Academy Awards Oscar for
Original Dramatic Score , and the 1971 BAFTA Anthony Asquith Award for Film Music , both to Michel Legrand . It counted among its fans Stanley Kubrick , who had
the film play on a television in a scene in The Shining .
</item 2>

<item 3>
The Deer Hunter
and most controversial , major theatrical films to be critical of the American involvement in Vietnam following 1975 when the war officially ended . While the
film opened the same year as Hal Ashby 's Coming Home , Sidney Furie 's The Boys in Company C , and Ted Post 's Go Tell the Spartans , it was the first film
about Vietnam to reach a wide audience and critical acclaim , culminating in the winning of the Oscar for Best Picture .
</item 3>

<item 4>
The Deer Hunter
Michael Deeley , Michael Cimino , John Peverall Nominated BAFTA Award for Best Direction Michael Cimino Nominated BAFTA Award for Best Actor in a Leading Role
Robert De Niro Nominated BAFTA Award for Best Actress in a Leading Role Meryl Streep Nominated BAFTA Award for Best Original Screenplay Deric Washburn , Michael
Cimino , Louis Garfinkle , Quinn K. Redeker Nominated BAFTA Award for Best Actor in a Supporting Role Christopher Walken Nominated BAFTA Award for Best Editing
Peter Zinner Won BAFTA Award for Best Cinematography Vilmos Zsigmond Won BAFTA Award for Best Sound Richard Portman , William McCaughey , Aaron Rochin , Darin
Knight Nominated Directors Guild of America Awards 1978 Directors Guild of America Award for Outstanding Directing -- Feature Film Michael Cimino Won 36th
Golden Globe Awards Golden Globe Award for Best Motion Picture -- Drama Barry Spikings Nominated Golden Globe Award for Best Director Michael Cimino Won Golden
Globe Award for Best Actor -- Motion Picture Drama Robert De Niro Nominated Golden Globe Award for Best Screenplay Deric Washburn Nominated Golden Globe Award
for Best Supporting Actor -- Motion Picture Christopher Walken
</item 4>

<item 5>
The Deer Hunter
( 1968 ) Midnight Cowboy ( 1969 ) Patton ( 1970 ) The French Connection ( 1971 ) The Godfather ( 1972 ) The Sting ( 1973 ) The Godfather Part II ( 1974 ) One
Flew Over the Cuckoo 's Nest ( 1975 ) 1976 -- 2000 Rocky ( 1976 ) Annie Hall ( 1977 ) The Deer Hunter ( 1978 ) Kramer vs. Kramer ( 1979 ) Ordinary People ( 1980
) Chariots of Fire ( 1981 ) Gandhi ( 1982 ) Terms of Endearment ( 1983 ) Amadeus ( 1984 ) Out of Africa ( 1985 ) Platoon ( 1986 ) The Last Emperor ( 1987 ) Rain
Man ( 1988 ) Driving Miss Daisy ( 1989 ) Dances with Wolves ( 1990 ) The Silence of the Lambs ( 1991 ) Unforgiven ( 1992 ) Schindler 's List ( 1993 ) Forrest
Gump ( 1994 ) Braveheart ( 1995 ) The English Patient ( 1996 ) Titanic (
</item 5>

<item 6>
The Deer Hunter
In addition , the film was nominated in four other categories : Best Actor in a Leading Role -- Robert De Niro ( lost to Jon Voight for Coming Home ) Best
Actress in a Supporting Role -- Meryl Streep ( lost to Maggie Smith for California Suite ) Best Cinematography -- Vilmos Zsigmond ( lost to Néstor Almendros for
Days of Heaven ) Best Writing , Screenplay Written Directly for the Screen -- Michael Cimino , Deric Washburn , Louis Garfinkle and Quinn Redeker ( lost to
Robert C. Jones , Waldo Salt and Nancy Dowd for Coming Home ) Golden globes ( edit ) Cimino won the film 's only Golden Globe Award for Best Director . Other
nominations the film included Best Motion Picture -- Drama , De Niro for Best Motion Picture Actor -- Drama , Walken for Best Motion Picture Actor in a
Supporting Role , Streep for Best Motion Picture Actress in a Supporting Role , and Washburn for Best Screenplay -- Motion Picture . Complete list of awards (
edit ) Award Recipients and nominees Result 51st Academy Awards Academy Award for Best Picture Barry Spikings , Michael Deeley ,
</item 6>

<item 7>
49th Academy Awards
USA -- Barbara Kopple Hollywood on Trial -- David Helpern Off the Edge -- Michael Firth People of the Wind -- Anthony Howarth and David Koff Volcano : An
Inquiry into the Life and Death of Malcolm Lowry -- Donald Brittain and John Kramer Number Our Days -- Lynne Littman and Barbara Myerhoff American Shoeshine
Blackwood The End of the Road Universe Best Live Action Short Film Best Animated Short Film In the Region of Ice -- Peter Werner Kudzu -- Marjorie Anne Short
The Morning Spider -- Julian Chagrin and Claude Chagrin Nightlife -- Claire Wilbur and Robin Lehman Number One -- Dyan Cannon and Vince Cannon Leisure --
Suzanne Baker Dedalo -- Manfredo Manfredi The Street -- Caroline Leaf and Guy Glover Best Original Score Best Original Song Score or Adaptation Score The Omen
-- Jerry Goldsmith Obsession -- Bernard Herrmann ( posthumous nomination ) The Outlaw Josey Wales -- Jerry Fielding Taxi Driver -- Bernard Herrmann ( posthumous
nomination ) Voyage of the Damned -- Lalo Schifrin Bound for Glory -- Leonard Rosenman Bugsy Malone -- Paul Williams A Star Is Born -- Roger Kellaway Best
Original Song Best Sound ``
</item 7>

<item 8>
Gerald Ford
Buckley Ronald Reagan Harold Stassen Third party and independent candidates American Party Nominee Thomas J. Anderson American Independent Party Nominee Lester
Maddox Communist Party Nominee Gus Hall VP nominee Jarvis Tyner Libertarian Party Nominee Roger MacBride VP nominee David Bergland People 's Party Nominee
Margaret Wright VP nominee Benjamin Spock Prohibition Party Nominee Ben Bubar VP nominee Earl Dodge Socialist Workers Party Nominee Peter Camejo VP nominee
Willie Mae Reid U.S. Labor Party Nominee Lyndon LaRouche Other 1976 elections : House Senate Gubernatorial Theodore Roosevelt Award winners 1967 : Eisenhower
1968 : Saltonstall 1969 : White 1970 : Hovde 1971 : Kraft Jr . 1972 : Holland 1973 : Bradley 1974 : Owens 1975 : Ford 1976 : Hamilton 1977 : Bradley 1978 :
Zornow 1979 : Chandler 1980 : Cooley 1981 : Linkletter 1982 : Cosby 1983 : Palmer 1984 : Lawrence 1985 : Fleming 1986 : Bush 1987 : Zable 1988 : Not presented
1989 : Ebert 1990 :
</item 8>

<item 9>
49th Academy Awards
The two events conflicted again five years later in 1982 . Contents ( hide ) 1 Winners and nominees 1.1 Special Achievement Award 1.2 Irving G. Thalberg
Memorial Award 1.3 Multiple nominations and awards 2 Presenters and performers 2.1 Presenters 2.2 Performers 3 See also 4 References Winners and Nominees ( edit
) Peter Finch , Best Actor winner Faye Dunaway , Best Actress winner Jason Robards , Best Supporting Actor winner Beatrice Straight , Best Supporting Actress
winner Winners are listed first , highlighted in boldface and indicated with a double dagger ( ) . Best Picture Best Director Rocky -- Robert Chartoff and Irwin
Winkler , producers All the President 's Men -- Walter Coblenz , producer Bound for Glory -- Robert F. Blumofe and Harold Leventhal , producers Network --
Howard Gottfried , producer Taxi Driver -- Julia Phillips and Michael Phillips , producers John G. Avildsen -- Rocky Alan J.
</item 9>

<item 10>
49th Academy Awards
Christian Szell Burt Young -- Rocky as Paulie Pennino Beatrice Straight -- Network as Louise Schumacher Jane Alexander -- All the President 's Men as Judy
Graham Hoback Jodie Foster -- Taxi Driver as Iris Steensma Lee Grant -- Voyage of the Damned as Lillian Rosen Piper Laurie -- Carrie as Margaret White Best
Screenplay Written Directly for the Screen Best Screenplay Based on Material from Another Medium Network -- Paddy Chayefsky Cousin Cousine -- Jean - Charles
Tacchella and Danièle Thompson The Front -- Walter Bernstein Rocky -- Sylvester Stallone Seven Beauties -- Lina Wertmüller All the President 's Men -- William
Goldman based on the book by Carl Bernstein and Bob Woodward Bound for Glory -- Robert Getchell based on the book by Woody Guthrie Fellini 's Casanova --
Federico Fellini and Bernardino Zapponi based on the autobiography Histoire de ma vie by Giacomo Casanova The Seven - Per - Cent Solution -- Nicholas Meyer
based on his novel Voyage of the Damned -- David Butler and Steve Shagan based on the book by Gordon Thomas and Max Morgan Witts Best Documentary Feature Best
Documentary Short Subject Harlan County , USA -- Barbara
</item 10>
=== END ITEM id=c1fe71ecfa790e26 ===
