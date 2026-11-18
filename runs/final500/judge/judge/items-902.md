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

=== ITEM id=935b0f4bea265a2c ===
Gold sentences and facts:
Sentence 0: R.H. Thomson as Matthew Cuthbert
  Fact 0.0: R.H. Thomson played the role of Matthew Cuthbert.

Retrieved items:
<item 1>
Graph query row:
actor: R.H. Thomson
relationship: RELATED_TO
</item 1>

<item 2>
Anne (TV series)
Jump up ^ `` Anne with an E : Season 1 '' . Metacritic . Retrieved 17 June 2017 . Jump up ^ Ashby , Emily . `` Anne with an E '' . Common Sense Media .
Retrieved 17 June 2017 . Jump up ^ Cerny , Tasha . `` ANNE WITH AN E Review : `` Episodes 1 - 3 '' `` . www.thetrackingboard.com . Retrieved 18 June 2017 . Jump
up ^ Inhat , Gwen . `` Anne With An E offers a winning , darker take on a familiar tale '' . A.V. Club . Retrieved 18 June 2017 . Jump up ^ Schneller , Johanna
( March 20 , 2017 ) . `` This Anne of Green Gables hints at miseries : What I 'm Watching '' . Toronto Star . access - date = requires url = ( help ) Jump up ^
Nguyen , Hanh . `` ' Anne with an E ' Review : A ' Breaking Bad ' Producer Updates ' Anne of Green Gables ' for Our Darker Times '' .
</item 2>

<item 3>
Anne (TV series)
Anne ( TV series ) - wikipedia Anne ( TV series ) Jump to : navigation , search Anne Also known as Anne with an E Genre Drama Created by Moira Walley - Beckett
Based on Anne of Green Gables by Lucy Maud Montgomery Screenplay by Moira Walley - Beckett Starring Amybeth McNulty Geraldine James R.H. Thomson Lucas Jade
Zumann Dalila Bela Corrine Koslo Aymeric Jett Montaz Opening theme `` Ahead by a Century '' by The Tragically Hip Country of origin Canada Original language ( s
) English No. of seasons No.
</item 3>

<item 4>
Anne (TV series)
Numeris . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 10 , 2017 ) . `` SUN onites SNET NHL TOR - COL 1181k VAN -- EDM 766k TSN CURL 1026k GLO NCISLA 1Mk
CBC ANNE 654k STORY US 546k CITY SHOTS 623k CTV SAVE HOPE 548k '' ( Tweet ) -- via Twitter . Jump up ^ Brioux , Bill ( @ BillBriouxTV ) ( April 24 , 2017 ) . ``
SJN o'nites SNet NHL TOR - WASH Game 6 3467k OTT - BOS 1284k OMNI2 TOR - WASH 121k CTV SAVE HOPE 686K CBC ANNE 656K STORY OF US 450K '' ( Tweet ) -- via Twitter
. Jump up ^ `` Anne with an E ( 2017 - ) '' . Rotten Tomatoes . Retrieved 4 January 2018 . Jump up ^ `` Anne with an E : Season 1 '' . Metacritic .
</item 4>

<item 5>
Anne (TV series)
When Matthew goes to pick the child up at the railway station , he finds not a boy , but a high - spirited and talkative girl , Anne Shirley . At first , the
Cuthberts are inclined to send her back , particularly after Marilla 's brooch goes missing , and Anne , in despair , runs away . Reconciled and settled in her
new home and school , Anne once again displays boundless enthusiasm which is nevertheless easily turned into despair when things go wrong , which they often do
. Slowly , her ebullient nature wins over those around her . Cast ( edit ) Main ( edit ) Amybeth McNulty as Anne Shirley Geraldine James as Marilla Cuthbert
R.H. Thomson as Matthew Cuthbert Dalila Bela as Diana Barry Lucas Jade Zumann as Gilbert Blythe Corrine Koslo as Rachel Lynde Aymeric Jett Montaz as Jerry
Baynard Others ( edit ) Jonathan Holmes as Mr. William Barry Helen Johns as Mrs. Eliza Barry Ryan Kiera Armstrong as Minnie Mary Barry Deborah Grover as
Josephine Barry Wayne Best as John Blythe Phillip Williams as Thomas Lynde David Ingram as Mr.
</item 5>

<item 6>
Anne (TV series)
of seasons No. of episodes 7 ( list of episodes ) Production Executive producer ( s ) Elizabeth Bradley Alex Sapot Sally Catto Debra Hayward Alison Owen Miranda
de Pencier Moira Walley - Beckett Producer ( s ) Susan Murdoch Cinematography Bobby Shore Running time 44 minutes Production company ( s ) Pelican Ballet
Northwood Entertainment Distributor Netflix Release Original network CBC Television Netflix ( worldwide ) Picture format 4K ( Ultra HD ) Audio format Dolby
Digital 5.1 with Descriptive Video Service track Original release March 19 , 2017 ( 2017 - 03 - 19 ) -- present ( present ) External links Website Anne ( titled
Anne with an E for international distribution on Netflix ) is a Canadian drama television series based on the 1908 novel Anne of Green Gables by Lucy Maud
Montgomery , and adapted by Emmy Award - winning writer and producer Moira Walley - Beckett . It airs on CBC in Canada , and elsewhere in the world it is
available for streaming on Netflix . The first season consists of seven episodes , with Niki Caro directing the 90 - minute season premiere .
</item 6>

<item 7>
Anne (TV series)
IndieWire . Retrieved 17 June 2017 . Jump up ^ Chaney , Jen . `` Anne of Green Gables Fans , You Will Love Netflix 's Anne With an E '' . Vulture.com . New York
Magazine . Retrieved 17 June 2017 . Jump up ^ Genzlinger , Neil ( May 11 , 2017 ) . `` Review : ' Anne With an E ' Is a Rewarding Return to Green Gables '' .
The New York Times . Retrieved 17 June 2017 . Jump up ^ Hirschlag , Annie . `` ' Anne with an E ' Review : The Neflix Adpatation Hits the Nostalgia Button and
More '' . Mic.com . Mic ( media Company ) . Retrieved 17 June 2017 . Jump up ^ Nawaz , Saleema ( March 21 , 2017 ) . `` We asked an Anne of Green Gables
superfan to review the first episode of CBC 's new TV series '' . Toronto Life . Retrieved 4 June 2017 .
</item 7>

<item 8>
Anne (TV series)
'' She went on to call Anne Shirley an `` accidental feminist '' , and how she `` really wanted to tell this story now . '' Broadcast and Release ( edit ) The
series initially premiered on March 19 , 2017 , on CBC and aired on a weekly basis , the season finale airing on April 30 , 2017 . The series debuted on Netflix
on May 12 , 2017 , under the title Anne with an E . Episodes ( edit ) Season 1 ( 2017 ) ( edit ) No . Title Directed by Written by Original air date Can .
viewers ( millions ) `` Your Will Shall Decide Your Destiny '' Niki Caro Moira Walley - Beckett March 19 , 2017 ( 2017 - 03 - 19 ) 0.999 When a miscommunication
brings a girl , Anne Shirley , to Green Gables instead of a boy , Marilla and Matthew Cuthbert are faced with a life - changing decision .
</item 8>

<item 9>
Anne (TV series)
7 `` Wherever You Are Is My Home '' Amanda Tapping Moira Walley - Beckett April 30 , 2017 ( 2017 - 04 - 30 ) N / A The Cuthberts vow to do whatever it takes to
save the farm which reminds Anne of the strength of friendship and love . Reception ( edit ) Critical response ( edit ) The series has achieved a rating of 88 %
on Rotten Tomatoes , the site 's critical consensus stating : `` Anne with an E uses its complex central character to offer a boldly stylish , emotionally
resonant spin on classic source material that satisfies in its own right . '' The series has received a rating of 79 on Metacritic based on fifteen reviews ,
indicating `` generally favorable reviews '' . Emily Ashby , writing for Common Sense Media , calls the series an `` exceptional '' and `` spectacular ''
interpretation , giving it four out of five stars . Tasha Cerny , contributor for the Tracking Board , praises the cinematography as lush and colourful , the
characters vibrant , and the plot `` surprisingly thrilling for a story about a young girl living in a small secluded community in the late nineteenth century .
</item 9>

<item 10>
Anne (TV series)
Writing for Entertainment Weekly , Isabella Beidenharn expressed similar feelings , but , `` putting the source material aside , it 's a fine show on its own ''
, and she conceded that `` inventing a dark side might help Anne With an E fit into today 's TV landscape '' . Allison Keene , writing for Collider , agrees
that Anne is a good drama on its own terms , but allows it is `` only a fair adaptation '' of the novel , at its best in the home scenes : `` Anne with an E is
undeniably the most stylish adaptation we 've ever seen of Anne of Green Gables . But its desire to reveal more of Anne 's miserable past in order to be more
true to what the desperation of an orphan is like feels at odds with Montgomery 's story . ''
</item 10>
=== END ITEM id=935b0f4bea265a2c ===

=== ITEM id=8890e811d8684d40 ===
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
1365 - 2486.2006. 01252. x . Jump up ^ Wilson , SK ; Dolman , AM ; Cheal , AJ ; Emslie , MJ Pratchett ; et al. ( 2009 ) . `` Maintenance of fish diversity on
disturbed coral reefs '' . Coral reefs. 28 ( 1 ) : 3 -- 14 . Bibcode : 2009CorRe ... 28 ... 3W . doi : 10.1007 / s00338 - 008 - 0431 - 2 . Jump up ^ Birkeland
and Lucas ( 1990 ) . Acanthaster planci : Major Management Problem of Coral Reefs . CRC Press . pp. 36 -- 40 . ISBN 0 - 8493 - 6599 - 6 . ^ Jump up to : Moran ,
PJ ( 1986 ) . `` The Acanthaster phenomenon '' . Oceanogr . Mar. Biol. Annu. Rev. 24 : 379 -- 480 .
</item 1>

<item 2>
Crown-of-thorns starfish
The spines , which are brittle , may also break off and become embedded in the tissue where they must be removed surgically . Saponins seem to occur throughout
the lifecycle of the crown - of - thorns starfish . The saponins in the eggs are similar to those in the adult tissues , and presumably these carry over to the
larvae . The mouthing behaviour of predators of juvenile starfish with rejection suggests the juveniles contain saponins . Behaviour ( edit ) Juveniles
concealed under coral rubble Feeding scars of white coral skeleton Feeding on branching Acropora coral Starfish ' competing ' for remaining live coral The adult
crown - of - thorns is a carnivorous predator that usually preys on reef coral polyps . It climbs onto a section of living coral colony using the large number
of tube feet on its oral surface and flexible body . It fits closely to the surface of the coral , even the complex surfaces of branching corals . It then
extrudes its stomach out through its mouth over the surface to virtually its own diameter . The stomach surface secretes digestive enzymes that allow the
starfish to absorb nutrients from the liquefied coral tissue .
</item 2>

<item 3>
Crown-of-thorns starfish
Crown - of - thorns starfish - wikipedia Crown - of - thorns starfish Jump to : navigation , search Crown - of - thorns starfish Scientific classification
Kingdom : Animalia Phylum : Echinodermata Class : Asteroidea Order : Valvatida Family : Acanthasteridae Genus : Acanthaster Species : Acanthaster planci
Binomial name Acanthaster planci ( Linnaeus , 1758 ) The crown - of - thorns starfish , Acanthaster planci , is a large , multiple - armed starfish that usually
preys upon hard , or stony , coral polyps ( Scleractinia ) . The crown - of - thorns starfish receives its name from venomous thorn - like spines that cover its
upper surface , resembling the biblical crown of thorns . It is one of the largest starfish in the world . A. planci has a very wide Indo - Pacific distribution
.
</item 3>

<item 4>
Crown-of-thorns starfish
ISBN 978 - 0 - 412 - 03541 - 8 . Jump up ^ Caso , M.J. ( 1974 ) . `` External morphology of Acanthaster planci ( Linnaeus ) '' . Journal of the Marine
Biological Associataion of India . 16 ( 1 ) : 83 -- 93 . Jump up ^ `` Crown of Thorns Starfish ( Acanthaster Planci ) . '' Crown of Thorns Starfish Videos ,
Photos and Facts . Web . 5 November 2014 . < http://www.arkive.org/crown-of-thorns-starfish/acanthaster-planci/ > . Jump up ^ Shedd , John G. ( 2006 ) . ``
Crown of Thorns Sea Star '' . Shedd Aquarium . Archived from the original on 2014 - 02 - 22 . Retrieved 2013 - 06 - 29 . ^ Jump up to : Lucas , J.S. ; Jones ,
M.M. ( 1976 ) .
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
00775630.167. 1 . Jump up ^ Kenchington , R ( 1978 ) . `` The crown - of - thorns crisis in Australia : A retrospective analysis '' . Environmental
Conservation. 5 ( 1 ) : 11 -- 20 . doi : 10.1017 / s0376892900005191 . Jump up ^ James , Peter ( 1976 ) . Requiem for the Reef . Brisbane : Foundation Press .
ISBN 0 - 9597383 - 0 - 4 . Jump up ^ Brown , and Willey ( 1972 ) . Crown of Thorns : The Death of the Great Barrier Reef ? . Angus and Robertson . ISBN 0 - 207
- 12390 - X . Jump up ^ Reichelt , RE ; Bradbury , RH ; Moran , P J. ( 1990 ) . `` The crown - of - thorns starfish , Acanthaster planci , on the Great Barrier
Reef '' . Mathematical and Computer Modelling.
</item 6>

<item 7>
Crown-of-thorns starfish
The spines are stiff and very sharp , and readily pierce through soft surfaces . Despite the battery of sharp spines on the aboral surface and blunt spines on
the oral surface , the crown - of - thorns starfish 's general body surface is membranous and soft . When the starfish is removed from the water , the body
surface ruptures and the body fluid leaks out , so the body collapses and flattens . The spines bend over and flatten , as well . They recover their shape when
reimmersed , if they are still alive . Taxonomy ( edit ) Family ( edit ) The family Acanthasteridae is monogeneric ; its position within the Asteroides is
unsettled . It is generally recognised as a distinctly isolated taxon . Recently , Blake concluded from comparative morphology studies of Acanthaster planci
that it has strong similarities with various members of the Oreasteridae . He transferred Acanthasteridae from the Spinulosida to the Valvatida and assigned it
a position near to the Oreasteridae , from which it appears to be derived .
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
ARKive : acanthaster - planci BioLib : 422188 EoL : 598484 Fossilworks : 100993 GBIF : 2271208 iNaturalist : 50915 ITIS : 157194 NCBI : 133434 WoRMS : 213289
NDL : 01030270 Retrieved from `` https://en.wikipedia.org/w/index.php?title=Crown-of-thorns_starfish&oldid=814983736 '' Categories : Acanthasteridae Fauna of
Western Australia Animals described in 1758 Invasive echinoderm species Hidden categories : All articles with dead external links Articles with dead external
links from September 2010 CS1 maint : Multiple names : authors list CS1 errors : external links Articles needing more detailed references Articles with '
species ' microformats All articles with unsourced statements Articles with unsourced statements from March 2016 Wikipedia external links cleanup from March
2016 Wikipedia spam cleanup from March 2016 Talk Contents About Wikipedia Wikispecies Afrikaans Azərbaycanca Български Cebuano Čeština Deutsch Español Français
한국어
</item 9>

<item 10>
Crown-of-thorns starfish
It occurs where coral reefs or hard coral communities occur in this region . Contents ( hide ) 1 Description 2 Taxonomy 2.1 Family 2.2 Genus and species 2.3
Genetic studies 3 Biology 3.1 Toxins 3.2 Behaviour 3.3 Predators 4 Life - cycle 4.1 Gametes and embryos 4.2 Larval stages 4.3 Metamorphosis , development and
growth 4.4 Field observations of life - cycle 5 Ecology 5.1 Ecological impact on reefs 5.2 Population outbreaks 5.3 Causes of population outbreaks 6 Population
control 7 References 8 External links Description ( edit ) Unusual brightly coloured crown - of - thorns starfish , Thailand The body form of the crown - of -
thorns starfish is fundamentally the same as that of a typical starfish , with a central disk and radiating arms . Its special traits , however , include being
disc - shaped , multiple - armed , flexible , prehensile , and heavily spined , and having a large ratio of stomach surface to body mass .
</item 10>
=== END ITEM id=8890e811d8684d40 ===

=== ITEM id=94b6b4f46e852fb5 ===
Gold sentences and facts:
Sentence 0: Dollree Mapp was an employee in the illegal gambling rackets dominated by Cleveland rackets kingpin Shon Birns . On May 23 , 1957 , police officers in Cleveland , Ohio , received an anonymous tip by phone that Virgil Ogletree , a numbers operator who was wanted for questioning in the bombing of rival numbers racketeer and future boxing promoter Don King 's home three days earlier , might be found at Mapp 's house , as well as illegal betting slips and equipment employed in the `` California Gold '' numbers operation set up by Mapp 's boyfriend Edward Keeling . Three officers went to the home and asked for permission to enter , but Mapp , after consulting her lawyer by telephone , refused to admit them without a search warrant . Two officers left , and one remained , watching the house from across the street .
  Fact 0.0: Dollree Mapp was an employee in the illegal gambling rackets.
  Fact 0.1: The illegal gambling rackets were dominated by Cleveland rackets kingpin Shon Birns.
  Fact 0.2: On May 23, 1957, police officers in Cleveland, Ohio, received an anonymous tip by phone.
  Fact 0.3: The anonymous tip said Virgil Ogletree might be found at Dollree Mapp's house.
  Fact 0.4: Virgil Ogletree was a numbers operator.
  Fact 0.5: Virgil Ogletree was wanted for questioning in the bombing of Don King's home three days before the tip.
  Fact 0.6: Don King was a rival numbers racketeer.
  Fact 0.7: Don King was a future boxing promoter.
  Fact 0.8: The anonymous tip said illegal betting slips might be found at Dollree Mapp's house.
  Fact 0.9: The anonymous tip said equipment employed in the "California Gold" numbers operation might be found at Dollree Mapp's house.
  Fact 0.10: The "California Gold" numbers operation was set up by Dollree Mapp's boyfriend Edward Keeling.
  Fact 0.11: Edward Keeling was Dollree Mapp's boyfriend.
  Fact 0.12: Three officers went to Dollree Mapp's home.
  Fact 0.13: The three officers asked for permission to enter the home.
  Fact 0.14: Dollree Mapp consulted her lawyer by telephone.
  Fact 0.15: Dollree Mapp refused to admit the officers without a search warrant.
  Fact 0.16: Two officers left.
  Fact 0.17: One officer remained, watching the house from across the street.

Retrieved items:
<item 1>
Mapp v. Ohio
23 , 1957 , p. 1 . Jump up ^ http://www.infoplease.com/us/supreme-court/cases/ar19.html Further reading ( edit ) Long , Carolyn ( 2006 ) . Mapp v. Ohio :
Guarding Against Unreasonable Searches and Seizures . University Press of Kansas . ISBN 0 - 7006 - 1441 - 9 . Stewart , Potter ( 1983 ) . `` The Road to Mapp v.
Ohio and beyond : The Origins , Development and Future of the Exclusionary Rule in Search - and - Seizure Cases '' . Columbia Law Review . 83 ( 6 ) : 1365 --
1404 . doi : 10.2307 / 1122492 . JSTOR 1122492 . Zotti , Priscilla H. Machado ( 2005 ) . Injustice for All : Mapp vs. Ohio and the Fourth Amendment . New York :
Peter Lang . ISBN 0 - 8204 - 7267 - 0 .
</item 1>

<item 2>
Mapp v. Ohio
Ohio Supreme Court reversed . Court membership Chief Justice Earl Warren Associate Justices Hugo Black Felix Frankfurter William O. Douglas Tom C. Clark John M.
Harlan II William J. Brennan , Jr . Charles E. Whittaker Potter Stewart Case opinions Majority Clark , joined by Warren , Black , Douglas , Brennan Concurrence
Black Concurrence Douglas Concurrence Stewart Dissent Harlan , joined by Frankfurter , Whittaker Laws applied U.S. Const . amends. IV , XIV This case overturned
a previous ruling or rulings Wolf v. Colorado ( 1949 ) Mapp v. Ohio , 367 U.S. 643 ( 1961 ) , was a landmark case in criminal procedure , in which the United
States Supreme Court decided that evidence obtained in violation of the Fourth Amendment , which protects against `` unreasonable searches and seizures , '' may
not be used in state law criminal prosecutions in state courts , as well as in federal criminal law prosecutions in federal courts as had previously been the
law .
</item 2>

<item 3>
Mapp v. Ohio
When asked about the warrant during oral argument at the Supreme Court , the Cleveland prosecutor arguing the case cautiously deflected the question , which the
court did not press . As the search of Mapp 's second - floor , 2 - bedroom apartment began , police handcuffed her for being belligerent . The police searched
the house thoroughly , and discovered Ogletree , who was subsequently cleared on the bombing charge , hiding in the apartment of the downstairs tenant , Minerva
Tate . In the search of Mapp 's apartment and in a footlocker in the basement of the house police found a quantity of `` California Gold '' betting slips and
paraphernalia . They also found a pistol and a small quantity of pornographic books and pictures which Mapp stated a previous tenant named Morris Jones had left
behind .
</item 3>

<item 4>
Mapp v. Ohio
Mapp v. Ohio - wikipedia Mapp v. Ohio Jump to : navigation , search Mapp v. Ohio Supreme Court of the United States Argued March 29 , 1961 Decided June 19 ,
1961 Full case name Dollree Mapp v. State of Ohio Citations 367 U.S. 643 ( more ) 81 S. Ct. 1684 ; 6 L. Ed . 2d 1081 ; 1961 U.S. LEXIS 812 ; 86 Ohio L. Abs. 513
; 16 Ohio Op . 2d 384 ; 84 A.L.R. 2d 933 Prior history Defendant convicted , Cuyahoga County , Ohio Court of Common Pleas ; affirmed , Ohio Court of Appeals ;
affirmed , Ohio Supreme Court 166 N.E. 2d 387 ( Ohio 1960 ) Subsequent history Rehearing denied , 368 U.S. 871 ( 1961 ) Holding The Fourth Amendment prohibition
against unreasonable searches and seizures , as applied to the states through the Fourteenth , excludes unconstitutionally obtained evidence from use in
criminal prosecutions . Ohio Supreme Court reversed .
</item 4>

<item 5>
Mapp v. Ohio
Supreme Court voted 5 - 3 in favor of Mapp . The Court overturned the conviction , and five justices found that the States were bound to exclude evidence seized
in violation of the 4th Amendment . This ruling officially applied the exclusionary rule to the states . See also ( edit ) United States Bill of Rights Fruit of
the poisonous tree Exclusionary rule References ( edit ) Jump up ^ Zotti , Priscilla . Injustice for All ( Peter Lang , 2005 ) . Jump up ^ Mapp v. Ohio , 367
U.S. 643 ( U.S. 1961 ) . Jump up ^ Duignan , Brian ( 2012 - 05 - 25 ) . `` Mapp v. Ohio '' . Encyclopædia Britannica Online . Encyclopædia Britannica Inc .
Retrieved March 19 , 2014 . Jump up ^ https://www.law.cornell.edu/supremecourt/text/367/643#writing - Jump up ^ Williams , Bob . `` Dolly Mapp Sheds Glamour Gal
Role '' , Cleveland Call and Post , Nov.
</item 5>

<item 6>
Mapp v. Ohio
The Supreme Court accomplished this by use of a principle known as selective incorporation ; in Mapp this involved the incorporation of the provisions , as
interpreted by the Court , of the Fourth Amendment which are applicable only to actions of the federal government into the Fourteenth Amendment due process
clause which is applicable to actions of the states . Contents ( hide ) 1 Circumstances of the case 2 Decision 3 See also 4 References 5 Further reading 6
External links Circumstances of the case ( edit ) This section needs additional citations for verification . Please help improve this article by adding
citations to reliable sources . Unsourced material may be challenged and removed . ( October 2017 ) ( Learn how and when to remove this template message )
Dollree Mapp was an employee in the illegal gambling rackets dominated by Cleveland rackets kingpin Shon Birns .
</item 6>

<item 7>
Mapp v. Ohio
Creighton ( 1987 ) State 42 U.S.C. § 1983 Monroe v. Pape ( 1961 ) Malley v. Briggs ( 1986 ) ( hide ) Incorporation against States Unreasonable search and
seizure Wolf v. Colorado ( 1949 ) Rochin v. California ( 1952 ) Irvine v. California ( 1954 ) Elkins v. United States ( 1960 ) Mapp v. Ohio ( 1961 ) Warrant
requirements Ker v. California ( 1963 ) Aguilar v. Texas ( 1964 ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=Mapp_v._Ohio&oldid=819300745 ''
Categories : United States Supreme Court decisions that overrule a prior Supreme Court decision Cuyahoga County , Ohio United States Supreme Court cases United
States Fourth Amendment case law 20th century American trials 1961 in United States case law Incorporation case law American Civil Liberties Union litigation
United States Supreme Court cases of the Warren Court Cleveland Division of Police Hidden categories : Articles needing additional references from October 2017
All articles needing additional references Talk Contents About Wikipedia Simple English Edit links This page was last edited on 8 January 2018 , at 16 : 31 .
</item 7>

<item 8>
Mapp v. Ohio
Mapp was arrested , charged , and cleared on a misdemeanor charge of possessing numbers paraphernalia ; but several months later , after she refused to testify
against Shon Birns , Edward Keeling and their associates at their trial that October for the attempted shakedown of Don King , she was prosecuted for possession
of the books , found guilty at a 1958 trial of `` knowingly having had in her possession and under her control certain lewd and lascivious books , pictures ,
and photographs in violation of 2905.34 of Ohio 's Revised Code '' , and sentenced to one to seven years in prison . Mapp then appealed her case to the Supreme
Court , on the grounds that the police had no probable cause to suspect her of having the books . She stated that the 4th Amendment should be incorporated to
the state and local level . She argued that the police could n't use the books as evidence in trial because they were found without a warrant and therefore
illegally recovered . Mapp is the first to call upon this law . Decision ( edit ) The U.S. Supreme Court voted 5 - 3 in favor of Mapp .
</item 8>

<item 9>
Mapp v. Ohio
On May 23 , 1957 , police officers in Cleveland , Ohio , received an anonymous tip by phone that Virgil Ogletree , a numbers operator who was wanted for
questioning in the bombing of rival numbers racketeer and future boxing promoter Don King 's home three days earlier , might be found at Mapp 's house , as well
as illegal betting slips and equipment employed in the `` California Gold '' numbers operation set up by Mapp 's boyfriend Edward Keeling . Three officers went
to the home and asked for permission to enter , but Mapp , after consulting her lawyer by telephone , refused to admit them without a search warrant . Two
officers left , and one remained , watching the house from across the street . Thirteen hours later , four cars full of police arrived and knocked on the door .
When she did n't answer , they forced the door . Mapp asked to see the alleged warrant and was shown a piece of paper which she snatched away from an officer ,
putting it inside her dress . The officers struggled with Mapp and recovered the piece of paper which was not seen by her or her lawyers again , and was not
introduced as evidence in any of the ensuing court proceedings .
</item 9>

<item 10>
Mapp v. Ohio
ISBN 0 - 8204 - 7267 - 0 . External links ( edit ) Wikisource has original text related to this article : Mapp v. Ohio Archival source documents relating to the
Mapp case at Cleveland Memory Dollree Mapp , Who Defied Police Search in Landmark Case , Is Dead - New York Times `` Supreme Court Landmark Case Mapp v. Ohio ''
from C - SPAN 's Landmark Cases : 12 Historic Supreme Court Decisions ( hide ) United States 4th Amendment case law Scope of the Fourth Amendment What
constitutes a search ? Ex parte Jackson ( 1878 ) Boyd v. United States ( 1886 ) Gouled v. United States ( 1921 ) Burdeau v. McDowell ( 1921 ) Hester v. United
States ( 1924 ) United States v. Lee ( 1927 ) Olmstead v. United States ( 1928 ) Goldman v. United States ( 1942 ) On Lee v. United States ( 1952 ) Abel v.
United States ( 1960 ) Silverman v. United States ( 1961 ) Lewis v.
</item 10>
=== END ITEM id=94b6b4f46e852fb5 ===

=== ITEM id=16b6b751bae71942 ===
Gold sentences and facts:
Sentence 0: `` Fore ! '' , originally an Australian interjection , is used to warn anyone standing or moving in the flight of a golf ball . The mention of the term in an 1881 Australian Golf Museum indicates that the term was in use at least as early as that period .
  Fact 0.0: Fore is originally an Australian interjection.
  Fact 0.1: Fore is used to warn anyone standing or moving in the flight of a golf ball.
  Fact 0.2: An 1881 Australian Golf Museum mention of the term exists.
  Fact 0.3: The 1881 mention indicates that the term Fore was in use at least as early as 1881.

Retrieved items:
<item 1>
Forensic firearm examination
Forensic firearm examination - wikipedia Forensic firearm examination Jump to : navigation , search Part of a series on Forensic science Physiological ( show )
Anthropology Biology Bloodstain pattern analysis Botany Dentistry DNA phenotyping DNA profiling Entomology Epidemiology Medicine Palynology Pathology Podiatry
Toxicology Social ( show ) Psychiatry Psychology Psychotherapy Social work Criminalistics ( hide ) Accounting Body identification Chemistry Facial
reconstruction Fingerprint analysis Firearm examination Footwear evidence Forensic arts Profiling Gloveprint analysis Palmprint analysis Questioned document
examination Vein matching Digital forensics ( show ) Computer exams Data analysis Database study Mobile devices Network analysis Photography Video analysis
Audio analysis Related disciplines ( show ) Electrical engineering Engineering Fire investigation Fire accelerant detection Fractography Linguistics Materials
engineering Polymer engineering Statistics Traffic collision reconstruction Related articles ( show ) Crime scene CSI effect Perry Mason syndrome Pollen
calendar Skid mark Trace evidence Use of DNA in forensic entomology Outline Forensic firearm examination is the forensic process of examining the
characteristics of firearms as well as any cartridges or bullets left behind at a crime scene . Specialists in this field are tasked with linking bullets and
cartridges to weapons and weapons to individuals .
</item 1>

<item 2>
Fore (golf)
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Fore_(golf)&oldid=802796028 '' Categories : Golf
terminology Scottish English Golf stubs Hidden categories : All articles with specifically marked weasel - worded phrases Articles with specifically marked
weasel - worded phrases from January 2016 All stub articles Talk About Wikipedia Hrvatski Italiano Norsk Srpskohrvatski / српскохрватски Svenska Edit links This
page was last edited on 28 September 2017 , at 14 : 56 . About Wikipedia
</item 2>

<item 3>
Pencil
Consequently , it was called plumbago ( Latin for `` lead ore '' ) . Because the pencil core is still referred to as `` lead '' , or a `` lead '' , many people
have the misconception that the graphite in the pencil is lead , and the black core of pencils is still referred to as lead , even though it never contained the
element lead . The words for pencil in German ( bleistift ) , Irish ( peann luaidhe ) , Arabic ( قلم رصاص qalam raṣāṣ ) , and some other languages literally
mean lead pen . The value of graphite would soon be realised to be enormous , mainly because it could be used to line the moulds for cannonballs ; the mines
were taken over by the Crown and were guarded . When sufficient stores of graphite had been accumulated , the mines were flooded to prevent theft until more was
required . The usefulness of graphite for pencils was discovered as well , but graphite for pencils had to be smuggled . The news of the usefulness of these
early pencils spread far and wide , attracting the attention of artists all over the known world . Because graphite is soft , it requires some form of
encasement .
</item 3>

<item 4>
Fore (golf)
Fore ( golf ) - Wikipedia Fore ( golf ) Jump to : navigation , search The tough rubber core of a golf ball makes it a hazard to others following a wayward shot
, despite its weight not exceeding 45.93 g ( 1.62 oz ) `` Fore ! '' , originally an Australian interjection , is used to warn anyone standing or moving in the
flight of a golf ball . The mention of the term in an 1881 Australian Golf Museum indicates that the term was in use at least as early as that period . It is
believed to come from the military `` beware before '' , which an artilleryman about to fire would yell alerting nearby infantrymen to drop to the ground to
avoid the shells overhead . ( Before may mean `` in front of ( the gun being fired ) '' ; fore may mean `` ( look ) ahead '' . ) Other possible origins include
the term being derived from the term `` fore - caddie '' , a caddie waiting down range from the golfer to find where the ball lands .
</item 4>

<item 5>
Orc (Middle-earth)
The original edition of The Hobbit and early drafts of The Lord of the Rings first used goblin everywhere and used hobgoblin for larger , more evil goblins .
While investigating possible sources for the word `` Hobbit '' , Tolkien realised he had made a mistake in using hob - , which is traditionally used to mean a
smaller entity , not a larger one . In later writings , after The Lord of the Rings , ( including The Adventures of Tom Bombadil and many essays published in
The Peoples of Middle - earth ) , Tolkien preferred the spelling Ork , evidently mainly to avoid the form Orcish , which would be naturally pronounced with the
c as / s / instead of / k / in English . Tolkien indeed used the adjective Orkish . According to George MacDonald Fraser in The Light 's on at Signpost ,
Tolkien answered a letter on this subject , confirming that Orcs and goblins were indeed identical . Physical appearance ( edit ) Orcs are described as ugly and
filthy fanged humanoids .
</item 5>

<item 6>
Forensic firearm examination
Class characteristics ( edit ) Preliminary examination of the bullet can exclude a large number of weapons by examining the general characteristics of a
recovered bullet . By determining general aspects of the fired ammunition , a number of weapons can be immediately excluded as being incapable of firing that
type of bullet . The make and model of the weapon can also be inferred from the combination of different class characteristics that are common to specific
manufactures . The three main class characteristics of all bullets are the lands and grooves , the caliber of the bullet , and the rifling twist . All three can
be tied directly to the type of barrel that was used to fire the bullet . The lands and grooves of barrel are the bumps and valleys created when the rifling is
created . The caliber is the diameter of the barrel . The twist is the direction of the striations left by the barrel 's rifling , clockwise ( right - handed )
or counterclockwise ( left - handed ) . Most barrels will have a right - handed twist with the exception of weapons created by the Colt 's Manufacturing Company
which uses left - handed twists .
</item 6>

<item 7>
Fore (golf)
These caddies were often warned about oncoming golf balls by a shout of the term `` fore - caddie '' which was eventually shortened to just `` fore ! '' . The
Colonel Bogey March is based on the descending minor third which the original Colonel Bogey whistled instead of yelling Fore around 1914 . Notes ( edit ) Jump
up ^ SND : Fore ^ Jump up to : Why Do Golfers Yell `` Fore '' for Errant Shots ? retrieved June 15 , 2007 Jump up ^ Derivation 4 Fore ! and Caddie retrieved
June 15 , 2007 Jump up ^ The Windsor Magazine : An Illustrated Monthly for Men and Women Vol. 33 , no . 192 ( December 1910 ) References ( edit ) Scott , James
Sibbald David , The British Army : Its Origin , Progress , and Equipment , 1868 Windsor Magazine : An Illustrated Monthly for Men and Women Vol. 32 , no . 292 (
December 1910 ) This golf - related article is a stub . You can help Wikipedia by expanding it .
</item 7>

<item 8>
Forensic firearm examination
1 Magnetic particle inspection 2.2. 2 Chemical restoration 3 Examination of cartridges 4 Examination of bullets 4.1 Class characteristics 4.2 Individual
characteristics 5 Striation databasing 6 Criticisms 7 See also 8 References History ( edit ) The ability to compare ammunition is a direct result of the
invention of rifling around the turn of the 16th century . By forcing the bullet to spin as it travels down the barrel of the weapon the bullet 's accuracy
greatly increases . At the same time , the rifling leaves marks on the bullet that are indicative of that particular barrel . Prior to mass production of
firearms , each barrel and bullet mold was hand made by gunsmiths making them unique . The first successful documented case of forensic firearm examination
occurred in 1835 when a member of the Bow Street Runners in London matched a recovered bullet from a murder victim to a specific mold in a suspect 's home
confirming that he made the bullet . Further evidence that the bullet maker was the perpetrator was found in his home and he was convicted .
</item 8>

<item 9>
Orc (Middle-earth)
Christopher Tolkien notes that while the author clearly differentiates between `` goblins and Orcs '' in the Tale of Tinúviel , the two terms appear to be
synonymous in the Tale of Turambar . The word Gongs is also used on a few occasions ; it appears both distinct from Orcs and as a sub-type of Orc . Christopher
Tolkien remarks that Gongs are `` evil beings obscurely related to Orcs '' . Both goblins and Orcs are occasionally mentioned as being `` of Melkor '' and also
acting independently . Orcs , goblins , and gongs appear in Tolkien 's two lexicons of elvish languages . The Qenya Lexicon from approximately 1915 defines Orc
as meaning `` monster , demon '' , and the Gnomish Lexicon dated 1917 defines Orc as `` goblin '' and Gong as `` one of a tribe of the Orcs , a goblin '' .
Christopher Tolkien also notes that in the latter lexicon , the word Gnome ( later Noldor ) is an emendation from Goblin .
</item 9>

<item 10>
Orc (Middle-earth)
About the goblins of The Hobbit , Tolkien wrote : They are not based on direct experience of mine ; but owe , I suppose , a good deal to the goblin tradition
... especially as it appears in George MacDonald , except for the soft feet which I never believed in . Middle - earth terms for Orcs ( edit ) In the High -
elven tongue Quenya , the word for `` Orc '' was urco , plural urqui , meaning `` bogey '' , or `` bogeyman '' , that is , something that provokes fear . In the
Grey - elven tongue Sindarin , it was orch , plural yrch . In the Dwarven tongue Khuzdul , it was rukhs , plural rakhâs . In the language of the Drúedain or
Wild Men , it was gorgûn . In the Black Speech of Mordor , the equivalent was Uruk , as can be seen in Uruk - hai , `` Orc - folk '' . Orc itself is from
Rohirric and the Hobbit - language , which shared linguistic roots , but the term is clearly related to the older Elvish words .
</item 10>
=== END ITEM id=16b6b751bae71942 ===

=== ITEM id=3fe4ead38d03bea0 ===
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
ATP synthase
Depiction of ATP synthase using the chemiosmotic proton gradient to power ATP synthesis through oxidative phosphorylation . In the 1960s through the 1970s ,
Paul Boyer , a UCLA Professor , developed the binding change , or flip - flop , mechanism theory , which postulated that ATP synthesis is dependent on a
conformational change in ATP synthase generated by rotation of the gamma subunit . The research group of John E. Walker , then at the MRC Laboratory of
Molecular Biology in Cambridge , crystallized the F catalytic - domain of ATP synthase . The structure , at the time the largest asymmetric protein structure
known , indicated that Boyer 's rotary - catalysis model was , in essence , correct . For elucidating this , Boyer and Walker shared half of the 1997 Nobel
Prize in Chemistry . The crystal structure of the F showed alternating alpha and beta subunits ( 3 of each ) , arranged like segments of an orange around a
rotating asymmetrical gamma subunit .
</item 1>

<item 2>
ATP synthase
This may have evolved to carry out the reverse reaction and act as an ATP synthase . In different species ( edit ) E coli ( edit ) E. coli ATP synthase is the
simplest known form of ATP synthase , with 8 different subunit types . Yeast ( edit ) Yeast ATP synthase is one of the best - studied eukaryotic ATP synthases ;
and five F , eight F subunits , and seven associated proteins have been identified . Most of these proteins have homologues in other eukaryotes . Plant ( edit )
In plants , ATP synthase is also present in chloroplasts ( CF F - ATP synthase ) . The enzyme is integrated into thylakoid membrane ; the CF - part sticks into
stroma , where dark reactions of photosynthesis ( also called the light - independent reactions or the Calvin cycle ) and ATP synthesis take place . The overall
structure and the catalytic mechanism of the chloroplast ATP synthase are almost the same as those of the mitochondrial enzyme . However , in chloroplasts , the
proton motive force is generated not by respiratory electron transport chain but by primary photosynthetic proteins .
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
The electrons are finally transferred to exogenous oxygen and , with the addition of two protons , water is formed . Efficiency of ATP production The table
below describes the reactions involved when one glucose molecule is fully oxidized into carbon dioxide . It is assumed that all the reduced coenzymes are
oxidized by the electron transport chain and used for oxidative phosphorylation . Step coenzyme yield ATP yield Source of ATP Glycolysis preparatory phase − 2
Phosphorylation of glucose and fructose 6 - phosphate uses two ATP from the cytoplasm .
</item 4>

<item 5>
Cellular respiration
The NADH pulls the enzyme 's electrons to send through the electron transport chain . The electron transport chain pulls H ions through the chain . From the
electron transport chain , the released hydrogen ions make ADP for an end result of 32 ATP . O attracts itself to the left over electron to make water . Lastly
, ATP leaves through the ATP channel and out of the mitochondria . Main article : Glycolysis Glycolysis is a metabolic pathway that takes place in the cytosol
of cells in all living organisms . This pathway can function with or without the presence of oxygen . In humans , aerobic conditions produce pyruvate and
anaerobic conditions produce lactate . In aerobic conditions , the process converts one molecule of glucose into two molecules of pyruvate ( pyruvic acid ) ,
generating energy in the form of two net molecules of ATP . Four molecules of ATP per glucose are actually produced , however , two are consumed as part of the
preparatory phase . The initial phosphorylation of glucose is required to increase the reactivity ( decrease its stability ) in order for the molecule to be
cleaved into two pyruvate molecules by the enzyme aldolase .
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
Graph query row:
entity: ATP synthase
relationship: LOCATED_IN
related: inner mitochondrial membrane
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
The overall reaction catalyzed by ATP synthase is : ADP + P + H ⇌ ATP + H O + H The formation of ATP from ADP and P is energetically unfavorable and would
normally proceed in the reverse direction . In order to drive this reaction forward , ATPase couples ATP synthesis during cellular respiration to an
electrochemical gradient created by the difference in proton ( H ) concentration across the mitochondrial membrane in eukaryotes or the plasma membrane in
bacteria . During photosynthesis in plants , ATP is synthesized by ATPase using a proton gradient created in the thylakoid lumen through the thylakoid membrane
and into the chloroplast stroma . ATP synthase consists of two main subunits , F and F , which has a rotational motor mechanism allowing for ATP production .
Because of its rotating subunit ATP Synthase is a molecular machine .
</item 10>
=== END ITEM id=3fe4ead38d03bea0 ===

=== ITEM id=914fb861bd6e89dd ===
Gold sentences and facts:
Sentence 0: Dual - Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) Age of Child Housing Food Transport . Clothing Health Child Care / Education Misc . Total Before - tax income : Less than $59,410 ( Average = $38,000 ) 0 to 2 2,990 1,160 1,170 640 630 2,040 420 9,050 3 to 5 2,990 1,260 1,230 500 590 1,910 620 9,100 6 to 8 2,990 1,710 1,350 570 660 1,290 630 8,760 9 to 11 2,990 1,970 1,350 580 710 1,910 630 9,520 12 to 14 2,990 2,130 1,480 690 1,090 1,110 700 9,960 15 to 17 2,990 2,120 1,630 730 1,010 1,290 580 9,970 Total 53,820 31,050 24,630 11,130 14,070 23,640 10,740 169,080 Before - tax income : $59,410 to $102,870 ( Average = $79,940 ) 0 to 2 3,920 1,405 1,690 760 850 2,860 890 12,370 3 to 5 3,920 1,490 1,740 610 800 2,740 1,090 12,390 6 to 8 3,920 2,100 1,860 680 940 1,680 1,110 12,290 9 to 11 3,920 2,400 1,870 710 1,000 2,110 1,100 13,110 12 to 14 3,920 2,580 1,990 840 1,410 1,910 1,170 13,820 15 to 17 3,920 2,570 2,150 900 1,330 2,400 1,050 14,320 Total 70,560 37,620 33,900 13,500 18,990 41,100 19,230 234,900 Before - tax income : More than $102,870 ( Average = $180,040 ) 0 to 2 7,100 1,900 2,550 1,050 980 5,090 1,790 20,460 3 to 5 7,100 2,000 2,610 880 930 4,970 1,990 20,480 6 to 8 7,100 2,630 2,730 970 1,080 3,910 2,000 20,420 9 to 11 7,100 2,980 2,730 1,010 1,150 4,350 2,000 21,320 12 to 14 7,100 3,190 2,860 1,170 1,610 4,700 2,070 22,700 15 to 17 7,100 3,180 3,020 1,280 1,520 6,460 1,950 24,510 Total 127,800 47,640 49,500 19,080 21,810 88,440 35,400 389,670
  Fact 0.0: The table gives USDA average spending per child for dual-parent families, not the basic cost of raising a child
  Fact 0.1: Spending categories are housing, food, transport, clothing, health, child care and education, miscellaneous, and total
  Fact 0.2: Dual-parent families with before-tax income under $59,410 have an average before-tax income of $38,000
  Fact 0.3: Dual-parent families with before-tax income under $59,410 and a child aged 0 to 2 spend $2,990 on housing, $1,160 on food, $1,170 on transport, $640 on clothing, $630 on health, $2,040 on child care and education, $420 on miscellaneous, totaling $9,050
  Fact 0.4: Dual-parent families with before-tax income under $59,410 and a child aged 3 to 5 spend $2,990 on housing, $1,260 on food, $1,230 on transport, $500 on clothing, $590 on health, $1,910 on child care and education, $620 on miscellaneous, totaling $9,100
  Fact 0.5: Dual-parent families with before-tax income under $59,410 and a child aged 6 to 8 spend $2,990 on housing, $1,710 on food, $1,350 on transport, $570 on clothing, $660 on health, $1,290 on child care and education, $630 on miscellaneous, totaling $8,760
  Fact 0.6: Dual-parent families with before-tax income under $59,410 and a child aged 9 to 11 spend $2,990 on housing, $1,970 on food, $1,350 on transport, $580 on clothing, $710 on health, $1,910 on child care and education, $630 on miscellaneous, totaling $9,520
  Fact 0.7: Dual-parent families with before-tax income under $59,410 and a child aged 12 to 14 spend $2,990 on housing, $2,130 on food, $1,480 on transport, $690 on clothing, $1,090 on health, $1,110 on child care and education, $700 on miscellaneous, totaling $9,960
  Fact 0.8: Dual-parent families with before-tax income under $59,410 and a child aged 15 to 17 spend $2,990 on housing, $2,120 on food, $1,630 on transport, $730 on clothing, $1,010 on health, $1,290 on child care and education, $580 on miscellaneous, totaling $9,970
  Fact 0.9: Dual-parent families with before-tax income under $59,410 have totals across all age groups of $53,820 housing, $31,050 food, $24,630 transport, $11,130 clothing, $14,070 health, $23,640 child care and education, $10,740 miscellaneous, and $169,080 overall
  Fact 0.10: Dual-parent families with before-tax income of $59,410 to $102,870 have an average before-tax income of $79,940
  Fact 0.11: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 0 to 2 spend $3,920 on housing, $1,405 on food, $1,690 on transport, $760 on clothing, $850 on health, $2,860 on child care and education, $890 on miscellaneous, totaling $12,370
  Fact 0.12: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 3 to 5 spend $3,920 on housing, $1,490 on food, $1,740 on transport, $610 on clothing, $800 on health, $2,740 on child care and education, $1,090 on miscellaneous, totaling $12,390
  Fact 0.13: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 6 to 8 spend $3,920 on housing, $2,100 on food, $1,860 on transport, $680 on clothing, $940 on health, $1,680 on child care and education, $1,110 on miscellaneous, totaling $12,290
  Fact 0.14: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 9 to 11 spend $3,920 on housing, $2,400 on food, $1,870 on transport, $710 on clothing, $1,000 on health, $2,110 on child care and education, $1,100 on miscellaneous, totaling $13,110
  Fact 0.15: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 12 to 14 spend $3,920 on housing, $2,580 on food, $1,990 on transport, $840 on clothing, $1,410 on health, $1,910 on child care and education, $1,170 on miscellaneous, totaling $13,820
  Fact 0.16: Dual-parent families with before-tax income of $59,410 to $102,870 and a child aged 15 to 17 spend $3,920 on housing, $2,570 on food, $2,150 on transport, $900 on clothing, $1,330 on health, $2,400 on child care and education, $1,050 on miscellaneous, totaling $14,320
  Fact 0.17: Dual-parent families with before-tax income of $59,410 to $102,870 have totals across all age groups of $70,560 housing, $37,620 food, $33,900 transport, $13,500 clothing, $18,990 health, $41,100 child care and education, $19,230 miscellaneous, and $234,900 overall
  Fact 0.18: Dual-parent families with before-tax income over $102,870 have an average before-tax income of $180,040
  Fact 0.19: Dual-parent families with before-tax income over $102,870 and a child aged 0 to 2 spend $7,100 on housing, $1,900 on food, $2,550 on transport, $1,050 on clothing, $980 on health, $5,090 on child care and education, $1,790 on miscellaneous, totaling $20,460
  Fact 0.20: Dual-parent families with before-tax income over $102,870 and a child aged 3 to 5 spend $7,100 on housing, $2,000 on food, $2,610 on transport, $880 on clothing, $930 on health, $4,970 on child care and education, $1,990 on miscellaneous, totaling $20,480
  Fact 0.21: Dual-parent families with before-tax income over $102,870 and a child aged 6 to 8 spend $7,100 on housing, $2,630 on food, $2,730 on transport, $970 on clothing, $1,080 on health, $3,910 on child care and education, $2,000 on miscellaneous, totaling $20,420
  Fact 0.22: Dual-parent families with before-tax income over $102,870 and a child aged 9 to 11 spend $7,100 on housing, $2,980 on food, $2,730 on transport, $1,010 on clothing, $1,150 on health, $4,350 on child care and education, $2,000 on miscellaneous, totaling $21,320
  Fact 0.23: Dual-parent families with before-tax income over $102,870 and a child aged 12 to 14 spend $7,100 on housing, $3,190 on food, $2,860 on transport, $1,170 on clothing, $1,610 on health, $4,700 on child care and education, $2,070 on miscellaneous, totaling $22,700
  Fact 0.24: Dual-parent families with before-tax income over $102,870 and a child aged 15 to 17 spend $7,100 on housing, $3,180 on food, $3,020 on transport, $1,280 on clothing, $1,520 on health, $6,460 on child care and education, $1,950 on miscellaneous, totaling $24,510
  Fact 0.25: Dual-parent families with before-tax income over $102,870 have totals across all age groups of $127,800 housing, $47,640 food, $49,500 transport, $19,080 clothing, $21,810 health, $88,440 child care and education, $35,400 miscellaneous, and $389,670 overall

Retrieved items:
<item 1>
Cost of raising a child
Lakh ) % of Cost Education 25.19 46 % Housing 10.40 19 % Entertainment 6.57 12 % Clothing 3.29 6 % Food 2.74 5 % Transportation 2.74 5 % Healthcare 2.19 4 %
Others 1.64 3 % Total 54.75 100 % Note : Estimate assumes cost of birth , but does n't consider any major illness in child . See also ( edit ) Only child Child
care Child support Childfree Cost of living Demographic economics Family economics Family planning Family values Single parent Middle class squeeze References (
edit ) Jump up ^ Shah , Anup ( September 20 , 2010 ) . `` Poverty Facts and Stats '' . Global Issues . Retrieved August 5 , 2012 . ^ Jump up to : State of the
World 's Children , 2005 , UNICEF Jump up ^ Martin Ravallion , Shaohua Chen and Prem Sangraula , Dollar a day revisited , World Bank , May 2008 Jump up ^ ``
Cost of a child LV = '' . www.lv.com .
</item 1>

<item 2>
Cost of raising a child
This statistic includes children . Based on UNICEF statistics , in developing countries the cost is roughly US $ 900 for raising a child for a year , and US
$16,200 for raising a child from birth to age 17 . ( This calculation is difficult , since families in developing countries often do not operate with currency ,
but barter or trade to provide for their children . ) Half of all children in the world live in poverty . United Kingdom ( edit ) The annual LV = ( Liverpool
Victoria ) Cost of a Child report calculates the cost of raising a child from birth to 21 years old . The latest report puts the cost as £ 231,843 . The Cost of
a child calculations , from birth to 21 years , were compiled by the Centre of Economic and Business Research ( CEBR ) for LV = in December 2015 and are based
on the cost for the 21 - year period to December 2015 . Additional research was conducted by Opinium Research LLP from 22 to 27 January 2016 . The total sample
size was 1,000 UK adults with children under the age of 18 and was conducted online .
</item 2>

<item 3>
Cost of raising a child
480 6 to 8 7,100 2,630 2,730 970 1,080 3,910 2,000 20,420 9 to 11 7,100 2,980 2,730 1,010 1,150 4,350 2,000 21,320 12 to 14 7,100 3,190 2,860 1,170 1,610 4,700
2,070 22,700 15 to 17 7,100 3,180 3,020 1,280 1,520 6,460 1,950 24,510 Total 127,800 47,640 49,500 19,080 21,810 88,440 35,400 389,670 All numbers are in US
dollars . Also notice that the dual parent and solo parent charts use different income ranges so that direct comparison between the two should be undertaken
with care .
</item 3>

<item 4>
Cost of raising a child
Cost of raising a child - wikipedia Cost of raising a child Jump to : navigation , search The cost of raising a child varies from country to country . The cost
of raising a child is usually determined according to a formula that accounts for major areas of expenditure , such as food , housing , and clothing . However ,
any given family 's actual expenses may differ from the estimates . For example , the rent on a home does not usually change when the tenants have another child
, so the family 's housing costs may remain the same . In other cases , the home may be too small , in which case the family might move to a larger home at a
higher cost . Contents ( hide ) 1 Developing countries 2 United Kingdom 3 United States 3.1 Dual - Parent Family - USDA Average Spending per Child ( not the
basic cost of raising ) 3.2 Single - Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) 4 India 5 See also 6 References 7
External links Developing countries ( edit ) According to Globalissues.org , `` Almost half the world -- over three billion people -- live on less than $2.50 a
day . ''
</item 4>

<item 5>
Cost of raising a child
970 1,350 580 710 1,910 630 9,520 12 to 14 2,990 2,130 1,480 690 1,090 1,110 700 9,960 15 to 17 2,990 2,120 1,630 730 1,010 1,290 580 9,970 Total 53,820 31,050
24,630 11,130 14,070 23,640 10,740 169,080 Before - tax income : $59,410 to $102,870 ( Average = $79,940 ) 0 to 2 3,920 1,405 1,690 760 850 2,860 890 12,370 3
to 5 3,920 1,490 1,740 610 800 2,740 1,090 12,390 6 to 8 3,920 2,100 1,860 680 940 1,680 1,110 12,290 9 to 11 3,920 2,
</item 5>

<item 6>
Cost of raising a child
www.lv.com . Retrieved 2016 - 06 - 27 . Jump up ^ Friday , H. Swint ( July 20 , 2007 ) . `` Cost of raising children not as high as government would have you
believe '' . caller.com . Retrieved March 31 , 2013 . Jump up ^ http://abcnews.go.com/Business/page?id=4019746 ^ Jump up to : Lino , Mark . `` Expenditures on
Children by Families , 2011 ( Miscellaneous Publication Number 1528 - 2011 ) '' ( PDF ) . United States Department of Agriculture , Center for Nutrition Policy
and Promotion . Retrieved August 5 , 2012 . Jump up ^ `` How much it cost to raise a child '' ( pdf ) . report . Economic Times . 18 April 2011 . Retrieved 13
July 2016 . External links ( edit ) `` Raising that ' 07 baby will cost $204,060 in U.S. '' - from Reuters , retrieved August 5 , 2012 .
</item 6>

<item 7>
Cost of raising a child
Berry Brazelton Tanya Byron Bryan Caplan Rudolf Dreikurs David Elkind Jo Frost Haim Ginott Thomas Gordon Alan E. Kazdin Truby King Annette Lareau Penelope Leach
Madeline Levine William Sears B.F. Skinner Benjamin Spock Baby and Child Care Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Cost_of_raising_a_child&oldid=788344537 '' Categories : Parenting Family Marriage Demographic economics Talk Contents
About Wikipedia Add links This page was last edited on 30 June 2017 , at 22 : 40 . About Wikipedia
</item 7>

<item 8>
Cost of raising a child
Single - Parent Family - USDA Average Spending per Child ( not the basic cost of raising ) ( edit ) Single - Parent Family - USDA Average Spending per Child (
not the basic cost of raising ) Age of Child Housing Food Transport . Clothing Health Child care / Education Misc . Total Before - tax income : Less than
$40,410 ( Average = $18,350 ) 0 to 2 2,840 1,400 680 410 520 1,400 510 7,760 3 to 5 2,840 1,370 920 330 600 1,940 610 8,610 6 to 8 2,840 1,830 1,030 340 670
1,940 780 8,450 9 to 11 2,840 2,010 1,060 400 620 1,360 740 9,030 12 to 14 2,840 2,150 1,130 420 940 1,120 840 9,440 15 to 17 2,840 2,270 1,
</item 8>

<item 9>
Cost of raising a child
Results have been weighted to nationally representative criteria United States ( edit ) The tax exemption per child is $3,900 in 2013 which is based on the
basic cost of raising a child . Plus there is a tax credit up to $1000 for each child. citation needed Based on a survey by the U.S. Department of Agriculture ,
the table below shows the estimated Average Spending on Children by Families . The data comes from the Consumer Expenditure Survey by the U.S. Department of
Labor , conducted from 2005 - 06 . The figures have been updated to 2011 dollars using the Consumer Price Index . However , some dispute the numbers as being
biased high for political reasons ( e.g. , Texas A&M University Finance Professor H. Swint Friday : `` The numbers , reported by the U.S. Department of
Agriculture , are outrageously misleading . Often government statistics are produced for political objectives that cause the research methodology to be biased
toward finding the highest dollar amount to support the objective . '' ) . These figures from the USDA go up to age 18 , and do not include any college or
university education .
</item 9>

<item 10>
Cost of raising a child
880 3,080 2,370 650 1,560 4,150 1,980 19,670 15 to 17 5,880 3,220 2,370 730 1,550 5,010 1,810 20,570 Total 105,840 48,390 40,140 10,830 22,410 72,810 33,000
333,420 All numbers are in US dollars . India ( edit ) Based on an estimate by Economic Times in April 2011 , the cost of raising a child from birth to age of
majority ( 21 Years ) for a middle to upper - middle income family comes to about ₹ 55 lakh ( US $87,000 ) in total . Cost break up is as follows : Cost of
raising a child in India ( as of April , 2011 ) Expenditure Class Estimated Cost ( Rs .
</item 10>
=== END ITEM id=914fb861bd6e89dd ===
