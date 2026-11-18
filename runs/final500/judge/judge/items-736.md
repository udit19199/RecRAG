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

=== ITEM id=b184b0c921620e67 ===
Gold sentences and facts:
Sentence 0: The United States two - dollar bill ( $2 ) is a current denomination of U.S. currency . The third U.S. President ( 1801 -- 09 ) , Thomas Jefferson , is featured on the obverse of the note . The reverse features an engraving of the painting The Declaration of Independence by John Trumbull . Throughout the $2 bill 's pre-1929 life as a large - sized note , it was issued as a United States Note , National Bank Note , silver certificate , Treasury or `` Coin '' Note and Federal Reserve Bank Note . When U.S. currency was changed to its current size , the $2 bill was issued only as a United States Note . Production went on until 1966 , when the series was discontinued . Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design . Two - dollar bills are seldom seen in circulation as a result of banking policies with businesses which has resulted in low production numbers due to lack of demand . This comparative scarcity in circulation , coupled with a lack of public knowledge that the bill is still in production and circulation , has also inspired urban legends and occasionally has created problems for people trying to use the bill to make purchases .
  Fact 0.0: The United States two-dollar bill ($2) is a current denomination of U.S. currency.
  Fact 0.1: Thomas Jefferson was the third U.S. President.
  Fact 0.2: Thomas Jefferson served as President from 1801 to 1809.
  Fact 0.3: Thomas Jefferson is featured on the obverse of the $2 bill.
  Fact 0.4: The reverse of the $2 bill features an engraving of the painting The Declaration of Independence by John Trumbull.
  Fact 0.5: Before 1929, the $2 bill was a large-sized note.
  Fact 0.6: Before 1929, the $2 bill was issued as a United States Note.
  Fact 0.7: Before 1929, the $2 bill was issued as a National Bank Note.
  Fact 0.8: Before 1929, the $2 bill was issued as a silver certificate.
  Fact 0.9: Before 1929, the $2 bill was issued as a Treasury or "Coin" Note.
  Fact 0.10: Before 1929, the $2 bill was issued as a Federal Reserve Bank Note.
  Fact 0.11: When U.S. currency was changed to its current size, the $2 bill was issued only as a United States Note.
  Fact 0.12: Production of the $2 bill went on until 1966, when the series was discontinued.
  Fact 0.13: Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design.
  Fact 0.14: Two-dollar bills are seldom seen in circulation as a result of banking policies with businesses.
  Fact 0.15: The banking policies with businesses resulted in low production numbers due to lack of demand.
  Fact 0.16: The comparative scarcity of $2 bills in circulation, coupled with a lack of public knowledge that the bill is still in production and circulation, has inspired urban legends.
  Fact 0.17: The comparative scarcity of $2 bills and the lack of public knowledge about them have occasionally created problems for people trying to use the bill to make purchases.

Retrieved items:
<item 1>
United States two-dollar bill
Between 1966 and 1976 , two - dollar notes were not printed . By 1869 , the $2 United States Note was redesigned with the now familiar portrait of Thomas
Jefferson to the left and a vignette of the United States Capitol in the center of the obverse . This note also featured green tinting on the top and left side
of the obverse . Although this note is technically a United States Note , TREASURY NOTE appeared on it instead of UNITED STATES NOTE . The reverse was
completely redesigned . This series was again revised in 1874 , changes on the obverse included removing the green tinting , adding a red floral design around
WASHINGTON D.C. , and changing the term TREASURY NOTE to UNITED STATES NOTE . The 1874 design was also issued as Series of 1875 and 1878 and by 1880 the red
floral design around WASHINGTON D.C. on the United States Note was removed and the serial numbers were changed to blue . This note with the red floral design
was also issued as Series of 1917 but with red serial numbers by that time .
</item 1>

<item 2>
United States two-dollar bill
1 Large size 3.3. 2 Small size 4 Visual snapshot 5 Usage 5.1 Lockheed U-2 5.2 Incidents 6 Uncut currency sheets 7 References 8 External links Denomination
overview ( edit ) The denomination of two dollars was authorized under a congressional act , and first used in March 1862 . The denomination was continuously
used until the 1960s ; by this time the United States Note was the only remaining class of U.S. currency the two - dollar bill was assigned to . In 1966 it was
decided to discontinue all United States Notes from production , which included the two - dollar bill . The two - dollar bill initially was not reassigned to
the Federal Reserve Note class of United States currency and was thus fully discontinued ; the Treasury Department cited the two - dollar bill 's low use and
unpopularity as the reason for not immediately resuming use of the denomination .
</item 2>

<item 3>
United States two-dollar bill
Although they are generally available at most banks , two - dollar bills are usually not handed out arbitrarily except upon specific request by the customer ,
and even then , may cause a delay with a trip to the vault . Rarity ( edit ) The seeming rarity of a $2 bill can be attributed to its low printing numbers as a
Federal Reserve Note . Hoarding of the series due to lack of public knowledge of the $2 bill has resulted in very few bills seen in circulation . After its
initial release , supplies of the Series 1976 two - dollar bill were allowed to dwindle until August 1996 when a new series dated 1995 began to be printed .
This series was only printed for the Federal Reserve Bank of Atlanta . Today , there is a common misconception by the general public that the $2 bill is no
longer being produced . According to the Treasury , it `` receives many letters asking why the $2 bill is no longer in circulation '' .
</item 3>

<item 4>
United States two-dollar bill
When U.S. currency was changed to its current size , the $2 bill was issued only as a United States Note . Production went on until 1966 , when the series was
discontinued . Ten years passed before the $2 bill was reissued as a Federal Reserve Note with a new reverse design . Two - dollar bills are seldom seen in
circulation as a result of banking policies with businesses which has resulted in low production numbers due to lack of demand . This comparative scarcity in
circulation , coupled with a lack of public knowledge that the bill is still in production and circulation , has also inspired urban legends and occasionally
has created problems for people trying to use the bill to make purchases . Contents ( hide ) 1 Denomination overview 2 Rarity 3 History 3.1 Large - sized notes
3.2 Small size notes 3.2. 1 1928 -- 1966 3.2. 2 1976 -- present 3.3 Series dates 3.3. 1 Large size 3.3.
</item 4>

<item 5>
Large denominations of United States currency
See history of the United States dollar . Passive retirement ( edit ) Although they are still legal tender in the United States , high - denomination bills were
last printed on December 27 , 1945 , and officially discontinued on July 14 , 1969 , by the Federal Reserve System , due to ' lack of use ' . The $5,000 and
$10,000 effectively disappeared well before then . The Federal Reserve began taking high - denomination currency out of circulation ( destroying large bills
received by banks ) in 1969 . As of May 30 , 2009 , only 336 $10,000 bills were known to exist ; 342 remaining $5,000 bills ; and 165,372 remaining $1,000 bills
. Due to their rarity , collectors often pay considerably more than the face value of the bills to acquire them . Some are in museums in other parts of the
world . For the most part , these bills were used by banks and the Federal government for large financial transactions . This was especially true for gold
certificates from 1865 to 1934 .
</item 5>

<item 6>
United States two-dollar bill
The note 's seal and serial numbers were red . The Series of 1928 $2 bill featured the treasury seal superimposed by the United States Note obligation to the
left and a large gray TWO to the right . In 1953 the $2 bill received minor design changes analogous to the $5 United States Note . The treasury seal was made
smaller and moved to the right side of the bill ; it was superimposed over the gray word TWO . The United States Note obligation now became superimposed over a
gray numeral 2 . The reverse remained unchanged . The final change to $2 United States Notes came in 1963 when the motto IN GOD WE TRUST was added to the
reverse over the Monticello . And , because dollar bills were soon to no longer be redeemable in silver , WILL PAY TO THE BEARER ON DEMAND was removed from the
obverse . Two - dollar bills , along with $5 and $100 United States Notes , were officially discontinued in August 1966 , although they remain legal tender .
1976 -- present ( edit ) Series 1976 first day of issue $2 bill with a canceled JFK postage stamp .
</item 6>

<item 7>
United States two-dollar bill
Retrieved February 9 , 2011 . CS1 maint : Uses authors parameter ( link ) Jump up ^ Stone , Suzanne J. ( March -- April 1976 ) . `` The $2 Bill Returns '' .
Retrieved February 9 , 2011 . Jump up ^ Bureau of Engraving and Printing . `` Annual Production Figures '' . Archived from the original on March 4 , 2007 .
Retrieved April 14 , 2007 . Jump up ^ `` Series 2009 $2 '' . USpapermoney.info . Retrieved April 29 , 2012 . Jump up ^ `` 2012 BEP Production Info via FOIA '' .
WheresGeorge.com . Retrieved April 29 , 2012 . Jump up ^ `` Series 2013 $2 '' . USpapermoney.info . Retrieved February 19 , 2014 . Jump up ^ `` 1869 $2 Legal
Tender Rainbow Note '' . usrarecurrency.com . Retrieved November 14 , 2015 . Jump up ^ Kathy Lawrence ( May 19 , 2011 ) .
</item 7>

<item 8>
United States two-dollar bill
Each note was an obligation of the issuing Federal Reserve Bank and could only be redeemed at the corresponding bank . The obverse of the note featured a
borderless portrait of Thomas Jefferson to left and wording in the entire center . The reverse featured a World War I battleship . Beginning in the 1950s ,
production of $2 bills began to decrease . The relative scarcity of the bills lead some to start saving any they found , with the inevitable result that the
notes became rarer in circulation . Currently , the circulation of $2 bills accounts for around 1 % of the U.S. currency in circulation . Small size notes (
edit ) ( 6.14 × 2.61 in ≅ 156 × 66 mm ) 1928 -- 1966 ( edit ) In 1928 , when all U.S. currency was changed to its current size , the $2 bill was issued only as
a United States Note . The obverse featured a cropped version of Thomas Jefferson 's portrait that had been on previous $2 bills . The reverse featured
Jefferson 's home , Monticello . The note 's seal and serial numbers were red .
</item 8>

<item 9>
United States two-dollar bill
In 1976 production of the two - dollar denomination was resumed and the two - dollar bill was finally assigned as a Federal Reserve Note , with a new reverse
design featuring John Trumbull 's depiction of the drafting of the United States Declaration of Independence replacing the previous design of Monticello . It
has remained a current denomination since then . It was estimated that if the two - dollar bill replaced approximately half of the one - dollar bills in
circulation , the federal government would be able to save about $26 million in 1976 dollars ( $112 million adjusted for inflation ) over the period from 1976
to 1981 , due to reduced production , storage , and shipping costs . However , due to their limited use , two - dollar bills are not printed as frequently in a
new series like other denominations , which are produced according to demand . Though some cash registers accommodate the bill , its slot is often used for
things like checks and rolls of coins . Some bill acceptors found in vending machines , self checkout lanes , and other automated kiosks are configured to
accommodate two - dollar bills , even if the fact is not stated on the label .
</item 9>

<item 10>
United States two-dollar bill
On April 13 , 1976 , the Treasury Department reintroduced the $2 bill as a cost - saving measure . Series 1976 $2 bills were redesigned and issued as a Federal
Reserve Note . The obverse design remains basically unchanged since 1928 and features the same portrait of Jefferson . A green treasury seal and serial numbers
replace the red used on the previous United States Notes . Since the reissue of the bill coincided with the United States Bicentennial , it was decided to use a
bicentennial themed design on the reverse . An engraved rendition ( not an exact reproduction ) of John Trumbull 's The Declaration of Independence replaced
Monticello on the reverse . First day issues of the new $2 bills could be taken to a post office and stamped with the date `` APR 13 1976 '' . In all ,
590,720,000 notes from Series 1976 were printed . Currently , stamped series 1976 $2 bills typically trade for about twice their face value . If the bills were
stamped in an exotic place , the value may be slightly higher .
</item 10>
=== END ITEM id=b184b0c921620e67 ===

=== ITEM id=57951c4cf0cfd93c ===
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
Physiologically , the function of the brain is to exert centralized control over the other organs of the body . The brain acts on the rest of the body both by
generating patterns of muscle activity and by driving the secretion of chemicals called hormones . This centralized control allows rapid and coordinated
responses to changes in the environment . Some basic types of responsiveness such as reflexes can be mediated by the spinal cord or peripheral ganglia , but
sophisticated purposeful control of behavior based on complex sensory input requires the information integrating capabilities of a centralized brain . The
operations of individual brain cells are now understood in considerable detail but the way they cooperate in ensembles of millions is yet to be solved . Recent
models in modern neuroscience treat the brain as a biological computer , very different in mechanism from an electronic computer , but similar in the sense that
it acquires information from the surrounding world , stores it , and processes it in a variety of ways . This article compares the properties of brains across
the entire range of animal species , with the greatest attention to vertebrates . It deals with the human brain insofar as it shares the properties of other
brains . The ways in which the human brain differs from other brains are covered in the human brain article .
</item 2>

<item 3>
Brain
In mammals , the cerebral cortex tends to show large slow delta waves during sleep , faster alpha waves when the animal is awake but inattentive , and chaotic -
looking irregular activity when the animal is actively engaged in a task . During an epileptic seizure , the brain 's inhibitory control mechanisms fail to
function and electrical activity rises to pathological levels , producing EEG traces that show large wave and spike patterns not seen in a healthy brain .
Relating these population - level patterns to the computational functions of individual neurons is a major focus of current research in neurophysiology .
Metabolism All vertebrates have a blood -- brain barrier that allows metabolism inside the brain to operate differently from metabolism in other parts of the
body . Glial cells play a major role in brain metabolism by controlling the chemical composition of the fluid that surrounds neurons , including levels of ions
and nutrients . Brain tissue consumes a large amount of energy in proportion to its volume , so large brains place severe metabolic demands on animals . The
need to limit body weight in order , for example , to fly , has apparently led to selection for a reduction of brain size in some species , such as bats .
</item 3>

<item 4>
Brain
Some aspects of brain structure are common to almost the entire range of animal species ; others distinguish `` advanced '' brains from more primitive ones , or
distinguish vertebrates from invertebrates . The simplest way to gain information about brain anatomy is by visual inspection , but many more sophisticated
techniques have been developed . Brain tissue in its natural state is too soft to work with , but it can be hardened by immersion in alcohol or other fixatives
, and then sliced apart for examination of the interior . Visually , the interior of the brain consists of areas of so - called grey matter , with a dark color
, separated by areas of white matter , with a lighter color . Further information can be gained by staining slices of brain tissue with a variety of chemicals
that bring out areas where specific types of molecules are present in high concentrations . It is also possible to examine the microstructure of brain tissue
using a microscope , and to trace the pattern of connections from one brain area to another . Cellular structure Neurons generate electrical signals that travel
along their axons . When a pulse of electricity reaches a junction called a synapse , it causes a neurotransmitter chemical to be released , which binds to
receptors on other cells and thereby alters their electrical activity .
</item 4>

<item 5>
Brain
PMID 7282965 . Jump up ^ Raichle , M ; Gusnard , DA ( 2002 ) . `` Appraising the brain 's energy budget '' . Proc. Natl. Acad. Sci. U.S.A. 99 ( 16 ) : 10237 --
10239 . Bibcode : 2002PNAS ... 9910237R . doi : 10.1073 / pnas. 172399499 . PMC 124895 . PMID 12149485 . Jump up ^ Mehagnoul - Schipper , DJ ; Van Der Kallen ,
BF ; Colier , WNJM ; Van Der Sluijs , MC ; Van Erning , LJ ; Thijssen , HO ; Oeseburg , B ; Hoefnagels , WH ; Jansen , RW ( 2002 ) . `` Simultaneous
measurements of cerebral oxygenation changes during brain activation by near - infrared spectroscopy and functional magnetic resonance imaging in healthy young
and elderly subjects '' . Hum Brain Mapp. 16 ( 1 ) : 14 -- 23 .
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
`` Acetate transport and utilization in the rat brain '' . J Neurochem. 109 Suppl 1 ( Suppl 1 ) : 46 -- 54 . doi : 10.1111 / j. 1471 - 4159.2009. 05895. x . PMC
2722917 . PMID 19393008 . Jump up ^ Soengas , JL ; Aldegunde , M ( 2002 ) . `` Energy metabolism of fish brain '' . Comparative Biochemistry and Physiology B.
131 ( 3 ) : 271 -- 296 . doi : 10.1016 / S1096 - 4959 ( 02 ) 00022 - 2 . PMID 11959012 . ^ Jump up to : Carew , TJ ( 2000 ) . `` Ch. 1 '' . Behavioral
Neurobiology : the Cellular Organization of Natural Behavior . Sinauer Associates . ISBN 978 - 0 - 87893 - 092 - 0 . Jump up ^ Dafny , N. `` Anatomy of the
spinal cord '' .
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
Norepinephrine , which is involved in arousal , comes exclusively from a nearby small area called the locus coeruleus . Other neurotransmitters such as
acetylcholine and dopamine have multiple sources in the brain , but are not as ubiquitously distributed as glutamate and GABA . Electrical activity Brain
electrical activity recorded from a human patient during an epileptic seizure . As a side effect of the electrochemical processes used by neurons for signaling
, brain tissue generates electric fields when it is active . When large numbers of neurons show synchronized activity , the electric fields that they generate
can be large enough to detect outside the skull , using electroencephalography ( EEG ) or magnetoencephalography ( MEG ) . EEG recordings , along with
recordings made from electrodes implanted inside the brains of animals such as rats , show that the brain of a living animal is constantly active , even during
sleep . Each part of the brain shows a mixture of rhythmic and nonrhythmic activity , which may vary according to behavioral state .
</item 9>

<item 10>
Brain
Jump up ^ Armstrong , E ( 1983 ) . `` Relative brain size and metabolism in mammals '' . Science . 220 ( 4603 ) : 1302 -- 1304 . Bibcode : 1983Sci ... 220.1302
A . doi : 10.1126 / science. 6407108 . PMID 6407108 . Jump up ^ Jerison , HJ ( 1973 ) . Evolution of the Brain and Intelligence . Academic Press . pp. 55 -- 74
. ISBN 978 - 0 - 12 - 385250 - 2 . Jump up ^ Parent , A ; Carpenter , MB ( 1995 ) . `` Ch. 1 '' . Carpenter 's Human Neuroanatomy . Williams & Wilkins . ISBN
978 - 0 - 683 - 06752 - 1 . Jump up ^ Pardridge , W ( 2005 ) . `` The Blood - Brain Barrier : Bottleneck in Brain Drug Development '' . NeuroRx. 2 ( 1 ) : 3 --
14 .
</item 10>
=== END ITEM id=57951c4cf0cfd93c ===

=== ITEM id=f57f2ccf31d5eaa0 ===
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
Subcutaneous injection - wikipedia Subcutaneous injection Jump to : navigation , search `` Subcut '' redirects here . For the song by Grand Mixer DXT featuring
Bill Laswell , see Aftermathematics . Play media 3D animation of a subcutaneous injection A subcutaneous injection is administered as a bolus into the subcutis
, the layer of skin directly below the dermis and epidermis , collectively referred to as the cutis . Subcutaneous injections are highly effective in
administering vaccines and medications such as insulin , morphine , diacetylmorphine and goserelin . Subcutaneous , as opposed to intravenous , injection of
recreational drugs is referred to as `` skin popping '' . Subcutaneous administration may be abbreviated as SC , SQ , sub-cu , sub-Q , SubQ , or subcut . Subcut
is the preferred abbreviation for patient safety . Subcutaneous tissue has few blood vessels and so drugs injected here are for slow , sustained rates of
absorption . It is slower than intramuscular injections but still faster than intradermal injections .
</item 1>

<item 2>
Gallbladder
This sits on a thin layer of connective tissue , the lamina propria . The mucosa is curved and collected into tiny outpouchings called rugae . A muscular layer
sits beneath the mucosa . This is formed by smooth muscle , with fibres that lie in longitudinal , oblique and transverse directions , and are not arranged in
separate layers . The muscle fibres here contract to expel bile from the gallbladder . A distinctive feature of the gallbladder is the presence of Rokitansky --
Aschoff sinuses , deep outpouchings of the mucosa that can extend through the muscular layer . The muscular layer is surrounded by a layer of connective and fat
tissue . The outer layer of the fundus of gallbladder , and the surfaces not in contact with the liver , are covered by a thick serosa , which is exposed to the
peritoneum . The serosa contains blood vessels and lymphatics . The surfaces in contact with the liver are covered in connective tissue . Variation ( edit ) The
gallbladder varies in size , shape , and position between different people .
</item 2>

<item 3>
Subcutaneous injection
25.2007. s112 . ( hide ) Routes of administration , dosage forms Oral Digestive tract ( enteral ) Solids Pill Tablet Capsule Pastille Time release technology
Osmotic delivery system ( OROS ) Liquids Decoction Elixir Electuary Emulsion Extended - release syrup Effervescent powder or tablet Herbal tea Hydrogel
Molecular encapsulation Powder Softgel Solution Suspension Syrup Syrup Concentrate for dilution and / or addition of carbonated water Tincture Buccal (
sublabial ) , sublingual Solids Orally disintegrating tablet ( ODT ) Film Lollipop Sublingual drops Lozenges Effervescent buccal tablet Chewing gum Liquids
Mouthwash Toothpaste Ointment Oral spray Respiratory tract Solids Smoking device Dry - powder inhaler ( DPI ) 0 0 Liquids Anaesthetic vaporizer Vaporizer
Nebulizer Metered - dose inhaler ( MDI ) Gas Oxygen mask and Nasal cannula Oxygen concentrator Anaesthetic machine Relative analgesia machine Ophthalmic ,
otologic ,
</item 3>

<item 4>
Subcutaneous injection
The 3 / 8 '' and 5 / 8 '' needles are most commonly used . Usually , no more than 1 mL of solution is given , compared to intradermal injections , where no more
than 0.5 mL is usually given . Procedure ( edit ) Subcutaneous injections are inserted at 45 to 90 degree angles , depending on amount of subcutaneous tissue
present and length of needle - a shorter , 3 / 8 '' needle is usually inserted 90 degrees and a 5 / 8 '' needle is usually inserted at 45 degrees . Medication
is administered slowly , about 10 seconds / milliliter . Subcutaneous insulin injections in diabetes mellitus ( edit ) An insulin pump with a subcutaneous
injection site . A person with insulin - dependent ( either type 1 or type 2 ) diabetes mellitus typically injects insulin subcutaneously . The injection should
be given under the skin , into the fat layer .
</item 4>

<item 5>
Subcutaneous injection
It is slower than intramuscular injections but still faster than intradermal injections . Contents ( hide ) 1 Injection sites 2 Equipment 3 Procedure 4
Subcutaneous insulin injections in diabetes mellitus 5 See also 6 References Injection sites ( edit ) Subcutaneous injection sites Sites include , The outer
area of the upper arm . The abdomen , from the rib margin to the iliac crest and avoiding a 2 - inch circle around the navel . This has the fastest rate of
absorption among the sites . The front of the thigh , midway to the outer side , 4 inches below the top of the thigh to 4 inches above the knee . This has a
slower rate of absorption than the upper arm . The upper back . The upper area of the buttock , just behind the hip bone . This has the slowest rate of
absorption among the sites . Equipment ( edit ) A 25 to 31 gauge thick , 3 / 8 '' to 1 '' long needle can be used . The size is determined by the amount of
subcutaneous tissue present , which is based on patient build .
</item 5>

<item 6>
Subcutaneous injection
otologic , nasal Nasal spray Ear drops Eye drops Ointment Hydrogel Nanosphere suspension Insufflation Mucoadhesive microdisc ( microsphere tablet ) Urogenital
Ointment Pessary ( vaginal suppository ) Vaginal ring Vaginal douche Intrauterine device ( IUD ) Extra-amniotic infusion Intravesical infusion Rectal ( enteral
) Ointment Suppository Enema Solution Hydrogel Murphy drip Nutrient enema Dermal Ointment Topical cream Topical gel Liniment Paste Film DMSO drug solution
Electrophoretic dermal delivery system Hydrogel Liposomes Transfersome vesicles Cream Lotion Lip balm Medicated shampoo Dermal patch Transdermal patch Contact (
rubbed into break in the skin ) Transdermal spray Jet injector Injection , infusion ( into tissue / blood ) Skin Intradermal Subcutaneous Transdermal implant
Organs Intracavernous Intravitreal Intra-articular injection Transscleral Central nervous system Intracerebral Intrathecal Epidural Circulatory ,
</item 6>

<item 7>
Serratus anterior muscle
The muscle is inserted along the medial border of the scapula between the superior and inferior angles along with being inserted along the thoracic vertebrae .
The muscle is divided into three named parts depending on their points of insertions : the serratus anterior superior is inserted near the superior angle the
serratus anterior intermediate is inserted along the medial border the serratus anterior inferior is inserted near the inferior angle . Relations ( edit ) The
serratus anterior lies deep to the subscapularis , from which it is separated by the subscapularis ( supraserratus ) bursa . It is separated from the rib by the
scapulothoracic ( infraserratus ) bursa . Nerve supply ( edit ) The serratus anterior is innervated by the long thoracic nerve ( Nerve of Bell ) , a branch of
the brachial plexus . The long thoracic nerve travels inferiorly on the surface of the serratus . The nerve is especially vulnerable during certain types of
surgery ( for example , during lymph node clearance from the axilla ( e.g.
</item 7>

<item 8>
Subcutaneous injection
The injection should be given under the skin , into the fat layer . See also ( edit ) Intramuscular injection Intravenous injection Intradermal injection
References ( edit ) Jump up ^ `` subcutaneous injection '' at Dorland 's Medical Dictionary Jump up ^ `` ISMP 's List of Error - Prone Abbreviations , Symbols ,
and Dose Designations '' ( PDF ) . www.ismp.org. 2013 . Retrieved 13 May 2013 . ^ Jump up to : Taylor , C.R. , Lillis , C. , LeMone , P. , Lynn , P. ( 2011 )
Fundamentals of nursing : The art and science of nursing care . Philadelphia : Lippincott Williams & Wilkins , page 749 . Jump up ^ `` Insulin Administration ''
. Diabetes Care . 25 ( suppl 1 ) : s112 -- s115 . January 2002 . doi : 10.2337 / diacare. 25.2007. s112 . ( hide ) Routes of administration ,
</item 8>

<item 9>
Subcutaneous injection
musculoskeletal Intravenous Intracardiac Intramuscular Intraosseous Intraperitoneal Nanocell injection Patient - Controlled Analgesia pump PIC line WikiProject
Pharmacy portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Subcutaneous_injection&oldid=827949994 '' Categories : Dosage forms Routes of
administration Injection ( medicine ) Talk Contents About Wikipedia Dansk Deutsch Español فارسی Français Italiano Nederlands Norsk Polski Português Slovenščina
Türkçe Українська Tiếng Việt 6 more Edit links This page was last edited on 27 February 2018 , at 18 : 16 . Text is available under the Creative Commons
Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a
registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 9>

<item 10>
Retina
The layers and anatomical correlation are as follows : OCT scan of the macular area of a retina at 800 nm , axial resolution 3 μm From innermost to outermost ,
the layers identifiable by OCT are as follows : # OCT Layer / Conventional Label Anatomical Correlate Reflectivity on OCT Specific anatomical boundaries ?
Additional references Posterior cortical vitreous Posterior cortical vitreous Hyper - reflective Yes Preretinal space In eyes where the vitreous has fully or
partially detached from the retina , this is the space created between the posterior cortical vitreous face and the internal limiting membrane of the retina .
Hypo - reflective Internal limiting membrane ( ILM ) Formed by Müller cell endfeet ( unclear if it can be observed on OCT ) Hyper - reflective No Nerve fiber
layer ( NFL ) Ganglion cell axons travelling towards the optic nerve Ganglion cell layer ( GCL ) Ganglion cell bodies ( and some displaced amacrine cells ) Hypo
- reflective 5 Inner plexiform layer ( IPL ) Synapses between bipolar , amacrine and ganglion cells Hyper - reflective 6 Inner nuclear layer ( INL ) a )
Horizontal ,
</item 10>
=== END ITEM id=f57f2ccf31d5eaa0 ===

=== ITEM id=1e41cbd694796f02 ===
Gold sentences and facts:
Sentence 0: Queen Elizabeth II is the sovereign , and her heir apparent is her eldest son , Charles , Prince of Wales . Next in line after him is Prince William , Duke of Cambridge , the Prince of Wales 's elder son . Third in line is Prince George , the eldest child of the Duke of Cambridge , followed by his sister , Princess Charlotte and younger brother , Prince Louis . Sixth in line is Prince Henry of Wales , the younger son of the Prince of Wales . Any of the first six in line marrying without the sovereign 's consent would be disqualified from succession .
  Fact 0.0: Queen Elizabeth II is the sovereign.
  Fact 0.1: Charles, Prince of Wales, is the heir apparent to the sovereign, Queen Elizabeth II.
  Fact 0.2: Charles, Prince of Wales, is the eldest son of Queen Elizabeth II.
  Fact 0.3: Prince William, Duke of Cambridge, is next in line for succession after Charles, Prince of Wales.
  Fact 0.4: Prince William is the elder son of the Prince of Wales.
  Fact 0.5: Prince George is third in line for succession.
  Fact 0.6: Prince George is the eldest child of the Duke of Cambridge.
  Fact 0.7: Princess Charlotte follows Prince George in the line of succession.
  Fact 0.8: Princess Charlotte is the sister of Prince George.
  Fact 0.9: Prince Louis is the younger brother of Prince George.
  Fact 0.10: Prince Henry of Wales is sixth in line for succession.
  Fact 0.11: Prince Henry of Wales is the younger son of the Prince of Wales.
  Fact 0.12: Any of the first six in line to the throne who marries without the sovereign's consent would be disqualified from succession.

Retrieved items:
<item 1>
Succession to the British throne
After Mary II died in 1694 , her husband continued to reign alone until his own death in 1702 . The line of succession provided for by the Bill of Rights was
almost at an end ; William and Mary never had any children , and Princess Anne 's children had all died . Therefore , Parliament passed the Act of Settlement .
The Act maintained the provision of the Bill of Rights whereby William would be succeeded by Princess Anne and her descendants , and thereafter by his own
descendants from future marriages . The Act , however , declared that they would be followed by James I & VI 's granddaughter Sophia , Electress Dowager of
Hanover ( the daughter of James 's daughter Elizabeth Stuart ) , and her heirs . As under the Bill of Rights , non-Protestants and those who married Roman
Catholics were excluded . Electress Sophia of Hanover ( 1630 -- 1714 ) Upon William 's death , Anne became Queen of England , Scotland and Ireland . Because the
Parliament of England settled on Sophia as Anne 's heir - presumptive without consulting Scottish leaders , the Estates of Scotland retaliated by passing the
Scottish Act of Security .
</item 1>

<item 2>
Succession to the British throne
Succession to the British throne - wikipedia Succession to the British throne This is the latest accepted revision , reviewed on 27 April 2018 . Jump to :
navigation , search The Sovereign 's Throne in the House of Lords , from which the speech is delivered at State Openings of Parliament Succession to the British
throne is determined by descent , gender ( for people born before October 2011 ) , legitimacy , and religion . Under common law , the Crown is inherited by a
sovereign 's children or by a childless sovereign 's nearest collateral line . The Bill of Rights 1689 and the Act of Settlement 1701 , restrict succession to
the throne to the legitimate Protestant descendants of Sophia of Hanover that are in `` communion with the Church of England '' . Spouses of Roman Catholics
were disqualified from 1689 until the law was amended in 2015 . Protestant descendants of those excluded for being Roman Catholics are eligible . Queen
Elizabeth II is the sovereign , and her heir apparent is her eldest son , Charles , Prince of Wales . Next in line after him is Prince William , Duke of
Cambridge , the Prince of Wales 's elder son .
</item 2>

<item 3>
Succession to the British throne
Edward died childless in 1972 . Edward 's abdication was `` a demise of the Crown '' ( in the words of the Act ) , and the Duke of York , his brother who was
then next in the line , immediately succeeded to the throne and to its `` rights , privileges , and dignities '' , taking the regnal name George VI . He in turn
was succeeded in 1952 by his own elder daughter , Elizabeth II . By that time the monarch of the United Kingdom no longer reigned in the greater part of Ireland
( which had become a republic in 1949 ) , but was the monarch of a number of independent sovereign states ( Commonwealth realms ) . Commonwealth realms ( edit )
Further information : Perth Agreement Elizabeth II at her coronation in 1953 , passing to the left of the Coronation Chair Each of the Commonwealth realms has
the same person as monarch and , to maintain that arrangement , they have agreed to continue the same line of succession ; some realms do so through domestic
succession laws , while others stipulate whoever is monarch of the United Kingdom will also be monarch of that realm .
</item 3>

<item 4>
Succession to the British throne
Each of those countries has the same person as monarch and the same order of succession . In 2011 , the prime ministers of the realms agreed unanimously to
adopt a common approach to amending the rules on the succession to their respective Crowns so that absolute primogeniture would apply for persons born after the
date of the agreement , instead of male - preference primogeniture , and the ban on marriages to Roman Catholics would be lifted , but the monarch would still
need to be in communion with the Church of England . After the necessary legislation had been enacted in accordance with each realm 's constitution , the
changes took effect on 26 March 2015 . Contents ( hide ) 1 Current line of succession 2 History 2.1 England 2.2 Scotland 2.3 After the union of the crowns 2.3.
1 Stuarts 2.3.
</item 4>

<item 5>
Succession to the British throne
Jane was also originally excluded on the premise that no woman could reign over England . Nonetheless , the will , which originally referred to Jane 's heirs -
male , was amended to refer to Jane and her heirs - male . Upon Edward VI 's death in 1553 , Jane was proclaimed Queen of England and Ireland . She was not
universally recognised and after nine days she was overthrown by the popular Mary . As Henry VIII 's will had been approved by an Act of Parliament in 1544 ,
Edward 's contravening will was unlawful and ignored . Mary was succeeded by her half - sister , Elizabeth , who broke with the precedents of many of her
predecessors , and refused to name an heir . Whilst previous monarchs ( including Henry VIII ) had specifically been granted authority to settle uncertain
successions in their wills , the Treasons Act 1571 asserted that Parliament had the right to settle disputes , and made it treason to deny Parliamentary
authority . Wary of threats from other possible heirs , Parliament further passed the Act of Association 1584 , which provided that any individual involved in
attempts to murder the Sovereign would be disqualified from succeeding .
</item 5>

<item 6>
The throne passes to Protestant, legitimate descendants of Sophia of Hanover under primogeniture; male preference applies only to those born before 28 October
2011. Adopted and out-of-wedlock children are ineligible, even if later legitimated. The 1772 Royal Marriages Act was repealed, but marriages voided under it
remain void for succession. Today, the first six in line need the sovereign’s consent to marry; without it, they and that marriage’s descendants are
disqualified.
</item 6>

<item 7>
Under the 1702 and 1703 Treason Acts, attempting to prevent the next heir from succeeding is treason, punishable by life imprisonment since 1998. In
Commonwealth realms, succession occurs immediately upon the sovereign’s death. The Accession Council then proclaims the new monarch, traditionally in public
places including London, York and Edinburgh. Accession Day is observed throughout the reign, with royal gun salutes in the UK. James I’s accession was notably
proclaimed by a council after Elizabeth I’s
</item 7>

<item 8>
Succession to the British throne
In February 1952 , on her accession , Elizabeth II was proclaimed as sovereign separately throughout her realms . In October 2011 , the heads of government of
all 16 realms agreed unanimously at a meeting held in Perth , Australia , to proposed changes to the royal succession laws that would see the order of
succession for people born after 28 October 2011 governed by absolute primogeniture -- wherein succession passes to an individual 's children according to birth
order , regardless of gender -- instead of male - preference primogeniture . They also agreed to lift a ban on those who marry Roman Catholics . The ban on
Catholics themselves was retained to ensure that the monarch would be in communion with the Church of England . The changes came into effect across the
Commonwealth realms on 26 March 2015 , after legislation was made according to each realm 's constitution . Following the changes coming into effect , the
positions of the first 27 in line remained unchanged , including Princess Anne and her children and grandchildren , until the birth of Princess Charlotte of
Cambridge on 2 May 2015 .
</item 8>

<item 9>
The British throne passes to legitimate Protestant descendants of Sophia of Hanover; Catholics remain excluded, though marrying a Catholic no longer
disqualifies an heir. Reforms agreed by the 16 Commonwealth realms took effect in 2015, ending male preference for those born after 28 October 2011 and
requiring sovereign consent for the first six heirs’ marriages. In April 2018, the first six were Charles, William, George, Charlotte, Louis and Harry.
Succession law evolved through Parliament,
</item 9>

<item 10>
Succession to the British throne
3 : `` A person who ( when the person marries ) is one of the 6 persons next in the line of succession to the Crown must obtain the consent of Her Majesty
before marrying . '' Jump up ^ Most sources , including Whitaker 's Almanack , exclude her as a confirmed Catholic , but Debrett 's includes her in line . Jump
up ^ Albert and Leopold Windsor were listed on The Official Website of the British Monarchy until 2015 and in the 2013 edition of Whitaker 's Almanack as
following Estella Taylor ( b 2004 ) and eligible to succeed ; MSN News , Debrett 's and Whitaker 's Almanack 2015 lists them after Lady Amelia Windsor and
before Lady Helen Taylor . They were baptised as Catholics , and are not listed in line in editions of Whitaker 's earlier than 2012 . Jump up ^ Neither the
Legitimacy Act 1926 nor the Legitimacy Act 1959 changed the Law of Succession . Jump up ^ Per the His Majesty 's Declaration of Abdication Act 1936 .
</item 10>
=== END ITEM id=1e41cbd694796f02 ===

=== ITEM id=fb8d0d2f1d37c91f ===
Gold sentences and facts:
Sentence 0: The video takes place with the band and the album 's two protagonists Christian ( Josh Boswell ) and Gloria ( Lisa Stelly ) taking refuge in a white room after robbing a bank . The police arrive outside the room and open fire through the window , to the couple 's terror . As bullets rain through the room , the band continues playing . Gloria picks up the phone and throws it into a fish tank . As the bullets continue to fly and tear apart the room , Christian and Gloria become calm and walk toward each other , unharmed by the bullets . They embrace and kiss as the room goes dark , recreating the 21st Century Breakdown cover art . As the song ends , the room lights up again , and they are still uninjured . After the room lights up , there are various shots of destruction in the room , including a shot of some of the writing on the walls . The writing includes an excerpt of the lyrics to `` 21 Guns '' , as well as those of the song `` See The Light '' which also appears on `` 21st Century Breakdown '' . The video may represent the meaning of the song in the 21st Century Breakdown story .
  Fact 0.0: The video features the band
  Fact 0.1: The video features the album's two protagonists, Christian and Gloria
  Fact 0.2: Christian is played by Josh Boswell
  Fact 0.3: Gloria is played by Lisa Stelly
  Fact 0.4: Christian and Gloria take refuge in a white room after robbing a bank
  Fact 0.5: The police arrive outside the room and open fire through the window
  Fact 0.6: The police fire to the couple's terror
  Fact 0.7: As bullets rain through the room, the band continues playing
  Fact 0.8: Gloria picks up the phone and throws it into a fish tank
  Fact 0.9: As the bullets continue to fly and tear apart the room, Christian and Gloria become calm
  Fact 0.10: Christian and Gloria walk toward each other, unharmed by the bullets
  Fact 0.11: Christian and Gloria embrace and kiss as the room goes dark
  Fact 0.12: The embrace and kiss recreate the 21st Century Breakdown cover art
  Fact 0.13: As the song ends, the room lights up again
  Fact 0.14: Christian and Gloria are still uninjured when the room lights up
  Fact 0.15: After the room lights up, there are various shots of destruction in the room
  Fact 0.16: The shots include a shot of some of the writing on the walls
  Fact 0.17: The writing includes an excerpt of the lyrics to 21 Guns
  Fact 0.18: The writing also includes lyrics of the song See The Light
  Fact 0.19: The song See The Light also appears on 21st Century Breakdown
  Fact 0.20: The video may represent the meaning of the song in the 21st Century Breakdown story

Retrieved items:
<item 1>
21 Guns (song)
GreenDay.com . November 13 , 2009 . Archived from the original on April 13 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` Press Release '' . GreenDay.com .
December 2 , 2009 . Archived from the original on June 8 , 2011 . Retrieved July 12 , 2010 . Jump up ^ Officially released video Green Day -- 21 Guns ( Cast
Version ) on YouTube Jump up ^ `` Get The New 21 Guns EP In Itunes Now '' . GreenDay.com . February 2 , 2010 . Archived from the original on June 8 , 2011 .
Retrieved July 12 , 2010 . Jump up ^ `` Australian-charts.com -- Green Day -- 21 Guns '' . ARIA Top 50 Singles . Retrieved October 27 , 2017 . Jump up ^ ``
Austriancharts.at -- Green Day -- 21 Guns '' ( in German ) . Ö3 Austria Top 40 . Retrieved October 27 , 2017 .
</item 1>

<item 2>
“21 Guns” is a Green Day rock ballad released as a 2009 single, with the original version running 5:21. Releases included “Favorite Son,” a Studio 880 version,
and a 4:37 radio edit; a cast version with the *American Idiot* musical cast runs 4:41. The song was written by Billie Joe Armstrong and Mike Dirnt, produced by
Butch Vig, and reached number one in Poland.
</item 2>

<item 3>
21 Guns (song)
The video was released February 1 . The second version of the song with the cast of American Idiot features Stark Sands , John Gallagher Jr. , and Michael Esper
singing their respective solos instead of Armstrong . It was performed with Armstrong singing lead male vocals at the 52nd Grammy Awards . Live EP 21 Guns Live
EP EP by Green Day Released September 4 , 2009 Recorded July 25 , 2009 in Albany , New York ; July 28 , 2009 Madison Square Garden , New York Genre Pop punk ,
punk rock , alternative rock Length 17 : 52 Label Reprise Producer Green Day Green Day released a live EP for the song including live performances of `` Welcome
to Paradise '' , `` Brain Stew / Jaded '' and `` F.O.D. '' recorded live in Albany , New York and Madison Square Garden . This EP was released on the Australian
iTunes store , Amazon UK , and Napster . Track listing Original version : Digital download / Maxi CD No . Title Length 1 . `` 21 Guns '' 5 : 21 2 .
</item 3>

<item 4>
The references document Green Day’s “21 Guns”: a GreenDay.com page (November 13, 2009), a press release (December 2, 2009), the official “Cast Version” video on
YouTube, and the 21 Guns EP’s iTunes release (February 2, 2010). They also cite the song’s Australian and Austrian chart listings, accessed October 27, 2017.
</item 4>

<item 5>
21 Guns (song)
British Phonographic Industry . Enter 21 Guns in the search field and then press Enter . Jump up ^ `` American single certifications -- Green Day -- 21 Guns ''
. Recording Industry Association of America . Retrieved October 13 , 2015 . If necessary , click Advanced , then click Format , then select Single , then click
SEARCH External links Lyrics of this song at MetroLyrics ( hide ) Green Day Billie Joe Armstrong Mike Dirnt Tré Cool Jason White John Kiffmeyer Studio albums 39
/ Smooth Kerplunk Dookie Insomniac Nimrod Warning American Idiot 21st Century Breakdown ¡ Uno ! ¡ Dos ! ¡ Tré ! Revolution Radio Live albums Live Tracks Tune In
, Tokyo ... Bullet in a Bible Awesome as Fuck Compilation albums 1,039 / Smoothed Out Slappy Hours International Superhits ! Shenanigans Green Day Demolicious
Greatest Hits : God 's Favorite Band EPs 1,000 Hours Slappy Sweet Children Last Night on Earth : Live in Tokyo Singles `` Longview '' `` Welcome to Paradise ''
`` Basket Case '' `` When I Come Around '' `` She '' `` J.A.R. '' `` Geek
</item 5>

<item 6>
21 Guns (song)
21 Guns ( song ) - wikipedia 21 Guns ( song ) Jump to : navigation , search `` 21 Guns '' Single by Green Day from the album 21st Century Breakdown Released May
25 , 2009 ( digital download ) July 14 , 2009 ( CD single ) Format Digital download , CD single , vinyl Recorded 2008 Genre Alternative rock Length 5 : 21 Label
Reprise Songwriter ( s ) Billie Joe Armstrong , David Bowie , Mike Dirnt , and Tré Cool Producer ( s ) Butch Vig and Green Day Green Day singles chronology ``
Know Your Enemy '' ( 2009 ) `` 21 Guns '' ( 2009 ) `` East Jesus Nowhere '' ( 2009 ) `` Know Your Enemy '' ( 2009 ) `` 21 Guns '' ( 2009 ) `` East Jesus Nowhere
'' ( 2009 ) Audio sample file help `` 21 Guns '' is a song by American punk rock band Green Day . It was released as the second single from their eighth album ,
21st Century Breakdown .
</item 6>

<item 7>
21 Guns (song)
The video may represent the meaning of the song in the 21st Century Breakdown story . Musical cast version `` 21 Guns '' Single by Green Day and the cast of
American Idiot from the album American Idiot : The Original Broadway Cast Recording B - side `` Like a Rolling Stone ( Bob Dylan Cover ) '' Released December 3
, 2009 Format Digital download , CD single Recorded 2009 Genre Alternative rock Length 4 : 41 Label Reprise Songwriter ( s ) Billie Joe Armstrong , David Bowie
, Mike Dirnt , and Tré Cool Producer ( s ) Green Day Green Day and the cast of American Idiot singles chronology `` 21 Guns '' ( 2009 ) `` When It 's Time '' (
2010 ) `` 21 Guns ( Musical Cast Version ) '' ( 2010 ) `` When It 's Time '' ( 2010 ) Green Day has recorded two versions of `` 21 Guns '' with the cast of
American Idiot . The recording was produced by Billie Joe Armstrong . The recording with Armstrong singing the male solo parts was released on December 3 to
Spinner.com and also released for digital download on December 22 .
</item 7>

<item 8>
“21 Guns” is a Green Day alternative-rock song from their eighth album, *21st Century Breakdown*, and its second single. Recorded in 2008 and produced by Butch
Vig and the band, it was released digitally on May 25, 2009, and on CD July 14. The song appeared in *The Vampire Diaries* and Green Day: Rock Band, and as
downloadable content for the *Rock Band* series. Marc Webb directed its Los Angeles-fil
</item 8>

<item 9>
21 Guns (song)
It became available for download on the music video game series Rock Band on July 7 , 2009 , along with the songs `` East Jesus Nowhere '' and `` Know Your
Enemy . The song was featured in episode 6 of The Vampire Diaries . The song is featured in the video game Green Day : Rock Band and is downloadable content for
the Rock Band series . A live version was also released on 21 Guns Live EP in September 2009 . Music video A music video was directed by Marc Webb and filmed in
Los Angeles on June 6 , 2009 . It premiered on MySpace on Monday , June 22 , 2009 at midnight EDT . A shorter version of the video also exists , which is set to
the radio edit of the song . It is the third Green Day video to feature touring guitarist Jason White performing with the band , following `` Wake Me Up When
September Ends '' and `` Working Class Hero '' . He is also seen briefly in the music videos for `` When I Come Around '' and `` Jesus of Suburbia '' but he is
not performing with the band .
</item 9>

<item 10>
21 Guns (song)
rockband.com . Archived from the original on April 25 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` 21 Guns Live EP '' . GreenDay.com . September 20 , 2009 .
Archived from the original on April 13 , 2010 . Retrieved July 12 , 2010 . Jump up ^ `` 21 Guns Video '' . Green Day.com . June 5 , 2009 . Archived from the
original on June 8 , 2011 . Retrieved July 12 , 2010 . Jump up ^ `` 2009 MTV Video Music Awards Winners '' . MTV . Retrieved July 12 , 2010 . Jump up ^ `` Green
Day -- making of 21 Guns video '' . MySpace Video . June 23 , 2009 . Archived from the original on July 2 , 2009 . Retrieved July 12 , 2010 . Jump up ^ `` Press
Release '' . GreenDay.com . November 13 , 2009 .
</item 10>
=== END ITEM id=fb8d0d2f1d37c91f ===

=== ITEM id=f4d065e81ca0bd5e ===
Gold sentences and facts:
Sentence 0: Ladies ' singles at the XXIII Olympic Winter Games Venue Gangneung Ice Arena Gangneung , South Korea Dates 21 -- 23 February Competitors 30 from 20 nations Winning score 239.57 Medalists 01 ! Alina Zagitova Olympic Athletes from Russia 02 ! Evgenia Medvedeva Olympic Athletes from Russia 03 ! Kaetlyn Osmond Canada ← 2014 2022 →
  Fact 0.0: The Ladies' singles event was held at the XXIII Olympic Winter Games.
  Fact 0.1: The Ladies' singles event was held at the Gangneung Ice Arena in Gangneung, South Korea.
  Fact 0.2: The Ladies' singles event was held on 21 to 23 February.
  Fact 0.3: The Ladies' singles event had 30 competitors from 20 nations.
  Fact 0.4: The winning score in the Ladies' singles event was 239.57.
  Fact 0.5: Alina Zagitova of Olympic Athletes from Russia was the first-place medalist.
  Fact 0.6: Evgenia Medvedeva of Olympic Athletes from Russia was the second-place medalist.
  Fact 0.7: Kaetlyn Osmond of Canada was the third-place medalist.
  Fact 0.8: The navigation links point to the previous event in 2014 and the next event in 2022.
Sentence 1: The ladies ' single figure skating competition of the 2018 Winter Olympics was held at the Gangneung Ice Arena in Gangneung , South Korea . The short program was held on 21 February , and the free skating was held on 23 February .
  Fact 1.0: The ladies' single figure skating competition was held at the 2018 Winter Olympics.
  Fact 1.1: The ladies' single figure skating competition was held at the Gangneung Ice Arena in Gangneung, South Korea.
  Fact 1.2: The short program was held on 21 February.
  Fact 1.3: The free skating was held on 23 February.

Retrieved items:
<item 1>
Graph query row:
event: Ladies' singles at the 2018 Winter Olympics
date: 21 -- 23 February
related_event: Free skating
related_date: 23 February
</item 1>

<item 2>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Contents ( hide ) 1 Records 2 Qualification 3 Schedule 4 Results 4.1 Short program 4.2 Free skating 4.3 Overall 5 References Records ( edit ) See also : World
and Olympic records set at the 2018 Winter Olympics For a complete list of figure skating records , see list of highest scores in figure skating . The following
new best scores were set during this competition : Component Skaters Score Date Ref Short program Evgenia Medvedeva ( OAR ) 81.61 21 February 2018 Alina
Zagitova ( OAR ) 82.92 The Olympic Athletes from Russia set two subsequent world records in the short program on February 21 , with Evgenia Medvedeva scoring
81.61 and then Alina Zagitova scoring 82.92 ten minutes later . Qualification ( edit ) Main article : Figure skating at the 2018 Winter Olympics --
Qualification A total of 30 skaters qualified to compete in the event , with each country allowed to only enter a maximum of three .
</item 2>

<item 3>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Pyeongchang 2018 Olympic Organizing Committee for the 2018 Winter Olympics . Retrieved 26 December 2017 . ^ Jump up to : `` Schedule '' . Retrieved 6 January
2018 . Jump up ^ Start list ^ Jump up to : Bieler , Des ; Janes , Chelsea ( 21 February 2018 ) . `` Alina Zagitova tops Russian teammate Evgenia Medvedeva 's
world - record short program score '' . The Washington Post . ISSN 0190 - 8286 . Retrieved 21 February 2018 . ^ Jump up to : `` Russia 's Zagitova beats
Medvedeva 's short program world record at Olympics '' . TASS. 21 February 2018 . Retrieved 21 February 2018 . Jump up ^ `` Qualification Systems for XXIII
Olympic Winter Games , PyeongChang 2018 , Figure skating '' . International Skating Union ( ISU ) . 17 May 2016 . Retrieved 20 January 2018 .
</item 3>

<item 4>
Figure skating at the 2018 Winter Olympics – Ladies' singles
17 May 2016 . Retrieved 20 January 2018 . Jump up ^ `` Extract from ISU Communication 2097 '' . www.iceskating.org.uk/ . National Ice Skating Association ( UK )
. Retrieved 20 January 2018 . Jump up ^ Short program results Jump up ^ Free skating results Jump up ^ Final results Olympic champions in figure skating --
Ladies ' singles 1908 : Madge Syers ( GBR ) 1920 : Magda Julin ( SWE ) 1924 : Herma Szabo ( AUT ) 1928 : Sonja Henie ( NOR ) 1932 : Sonja Henie ( NOR ) 1936 :
Sonja Henie ( NOR ) 1948 : Barbara Ann Scott ( CAN ) 1952 : Jeannette Altwegg ( GBR ) 1956 : Tenley Albright ( USA ) 1960 : Carol Heiss ( USA ) 1964 : Sjoukje
Dijkstra ( NED ) 1968 : Peggy Fleming ( USA ) 1972 : Trixi Schuba ( AUT ) 1976 : Dorothy
</item 4>

<item 5>
Figure skating at the 2018 Winter Olympics – Ladies' singles
S. Classic Lombardia Trophy Autumn Classic Ondrej Nepela Trophy Nebelhorn Trophy Finlandia Trophy Minsk - Arena Ice Star Warsaw Cup Tallinn Trophy Golden Spin
of Zagreb Junior internationals ISU Junior Grand Prix Junior Grand Prix Final National championships Canada China Germany Japan Russia South Korea United States
Retrieved from `` https://en.wikipedia.org/w/index.php?title=Figure_skating_at_the_2018_Winter_Olympics_ -- _Ladies % 27_singles&oldid = 827594424 '' Categories
: Figure skating at the 2018 Winter Olympics Women 's events at the 2018 Winter Olympics Talk Contents About Wikipedia Čeština Deutsch Français 한국어 Italiano
Magyar Nederlands Norsk Polski Português Română Русский Suomi Türkçe Українська 中文 7 more Edit links This page was last edited on 25 February 2018 , at 17 : 06
. About Wikipedia
</item 5>

<item 6>
Graph query row:
event: Ladies' singles at the 2018 Winter Olympics
date: 21 -- 23 February
related_event: 2018 Winter Olympics
related_date: 2018
</item 6>

<item 7>
Figure skating at the 2018 Winter Olympics – Ladies' singles
24 quotas were handed out during the 2017 World Figure Skating Championships and the remaining six were given out at the 2017 CS Nebelhorn Trophy . Each country
decided the entry of its teams , and athletes winning the quota were not necessarily granted the right to compete . All athletes competing must have met the
minimum total elements score , which does not include component scores . For the short program this was 20.00 and the free skate 36.00 . Schedule ( edit ) All
times are ( UTC + 9 ) . Date Time Round 21 February 10 : 00 Short program 23 February 10 : 00 Free skating Results ( edit ) Short program ( edit ) The short
program was held on 21 February . Pl . Name Nation TSS TES PCS SS TR PE CH IN Ded StN Alina Zagitova Olympic Athletes from Russia 82.92 WR 45.30 37.62 9.36 9.18
9.64 9.43 9.43 0.00 28 Evgenia Medvedeva Olympic Athletes from Russia 81.61 43.
</item 7>

<item 8>
Figure skating at the 2018 Winter Olympics – Ladies' singles
Figure skating at the 2018 Winter Olympics -- ladies ' singles - wikipedia Figure skating at the 2018 Winter Olympics -- ladies ' singles Jump to : navigation ,
search Ladies ' singles at the XXIII Olympic Winter Games Venue Gangneung Ice Arena Gangneung , South Korea Dates 21 -- 23 February Competitors 30 from 20
nations Winning score 239.57 Medalists 01 ! Alina Zagitova Olympic Athletes from Russia 02 ! Evgenia Medvedeva Olympic Athletes from Russia 03 ! Kaetlyn Osmond
Canada ← 2014 2022 → Figure skating at the 2018 Winter Olympics Qualification Singles men ladies Pairs mixed Ice dancing mixed Team event mixed The ladies '
single figure skating competition of the 2018 Winter Olympics was held at the Gangneung Ice Arena in Gangneung , South Korea . The short program was held on 21
February , and the free skating was held on 23 February .
</item 8>

<item 9>
Figure skating at the 2018 Winter Olympics – Ladies' singles
97 20 23 Ivett Tóth Hungary 150.43 53.22 23 97.21 23 24 Isadora Williams Brazil 144.18 55.74 17 88.44 24 Did not advance to free skating 25 Larkyn Austman
Canada N / A 51.42 25 N / A N / A 26 Diāna Ņikitina Latvia N / A 51.12 26 N / A N / A 27 Giada Russo Italy N / A 50.88 27 N / A N / A 28 Anita Östlund Sweden N
/ A 49.14 28 N / A N / A 29 Anna Khnychenkova Ukraine N / A 47.59 29 N / A N / A 30 Aiza Mambekova Kazakhstan N / A 44.40 30 N / A N / A References ( edit )
Jump up ^ `` Venues '' . www.pyeongchang2018.com/ . Pyeongchang 2018 Olympic Organizing Committee for the 2018 Winter Olympics .
</item 9>

<item 10>
Graph query row:
event: Ladies' singles at the 2018 Winter Olympics
date: 21 -- 23 February
related_event: short program
related_date: 21 February
</item 10>
=== END ITEM id=f4d065e81ca0bd5e ===
