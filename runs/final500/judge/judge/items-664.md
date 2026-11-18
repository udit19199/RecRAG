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

=== ITEM id=4f9dad478f40df78 ===
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
This exclusion is not affected by changes subsequent to the Perth Agreement . History ( edit ) This section needs additional citations for verification . Please
help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( May 2015 ) ( Learn how and when to
remove this template message ) The current succession law in the United Kingdom evolved from succession law in both England and Scotland . Originally in both
countries , there were no fixed rules governing succession to the throne . The individual could have relied on inheritance , statute , election ( by Parliament
or by another body ) , nomination ( by a reigning sovereign in his or her will ) , conquest or prescription ( de facto possession of the Crown ) . It was often
unclear which of these bases should take precedence ; often , the outcome depended not on the legal strength of the claims , but on the political or military
power of the claimants . However , over time , the default rule became male primogeniture : later monarchs coming to the throne by exception to this rule went
to great lengths to explain and justify going against these rules , and to prove their rivals illegitimate . Eventually , Parliament asserted control of
succession .
</item 1>

<item 2>
Succession to the British throne
1 Stuarts 2.3. 2 Hanoverians and Windsors 2.4 Commonwealth realms 3 Current rules 3.1 Marriages 3.2 Religion 3.3 Treason 4 Accession 5 See also 6 Notes 7
References 8 Sources 9 External links Current line of succession ( edit ) First six in line from 23 April 2018 1 . The Prince of Wales 2 . The Duke of Cambridge
3 . Prince George of Cambridge 4 . Princess Charlotte of Cambridge 5 . Prince Louis of Cambridge 6 . Prince Henry of Wales No official , complete version of the
line of succession is currently maintained . The exact number , in remoter collateral lines , of the persons who would be eligible is uncertain . In 2001
genealogist William Addams Reitwiesner compiled a list of 4,973 living descendants of the Electress Sophia in order of succession , but did so disregarding
Roman Catholic status . When updated in January 2011 , the number was more than five thousand .
</item 2>

<item 3>
Succession to the British throne
Succession to the British throne - wikipedia Succession to the British throne This is the latest accepted revision , reviewed on 27 April 2018 . Jump to :
navigation , search The Sovereign 's Throne in the House of Lords , from which the speech is delivered at State Openings of Parliament Succession to the British
throne is determined by descent , gender ( for people born before October 2011 ) , legitimacy , and religion . Under common law , the Crown is inherited by a
sovereign 's children or by a childless sovereign 's nearest collateral line . The Bill of Rights 1689 and the Act of Settlement 1701 , restrict succession to
the throne to the legitimate Protestant descendants of Sophia of Hanover that are in `` communion with the Church of England '' . Spouses of Roman Catholics
were disqualified from 1689 until the law was amended in 2015 . Protestant descendants of those excluded for being Roman Catholics are eligible . Queen
Elizabeth II is the sovereign , and her heir apparent is her eldest son , Charles , Prince of Wales . Next in line after him is Prince William , Duke of
Cambridge , the Prince of Wales 's elder son .
</item 3>

<item 4>
Succession to the British throne
A coronation is not necessary for a sovereign to reign ; for example , Edward VIII was never crowned , yet during his short reign was the undoubted king . See
also ( edit ) Biography portal Royalty portal United Kingdom portal Accession Declaration Act 1910 Alternative successions of the English crown British prince
British princess Family tree of the British royal family Jacobite succession List of descendants of George V List of heirs to the British throne List of heirs
to the English throne List of heirs of Scotland List of monarchs in the British Isles Patrilineal descent of Elizabeth II Posthumous birth Royal Succession Act
2013 ( New Zealand ) Royal Succession Bills and Acts Succession to the Crown Act 2013 ( United Kingdom ) Succession to the Crown Act 2015 ( Australia )
Succession to the Throne Act , 2013 ( Canada ) Perth Agreement 2011 and other articles , linked directly or indirectly -- Wikipedia book Notes ( edit ) Jump up
^ Succession to the Crown Act 2013 , s .
</item 4>

<item 5>
Succession to the British throne
The first to be affected by the changes , on the day they came into effect in March , were the children of Lady Davina Lewis -- her son Tāne ( born 2012 ) and
her daughter Senna ( born 2010 ) -- who were reversed in the order of succession , becoming 29th and 28th in line respectively . Current rules ( edit ) The Bill
of Rights and the Act of Settlement ( restated by the Acts of Union ) still govern succession to the throne . They were amended in the United Kingdom by the
Succession to the Crown Act 2013 , which was passed mainly `` to make succession to the Crown not depend on gender '' and `` to make provision about Royal
Marriages '' ( according to its long title ) , thereby implementing the Perth Agreement in the UK and in those realms that , by their laws , have as their
monarch automatically whoever is monarch of the UK . Other realms passed their own legislation . Anyone ineligible to succeed is treated as if they were `` dead
'' . That individual 's descendants are not also disqualified , unless they are personally ineligible .
</item 5>

<item 6>
Succession to the British throne
Third in line is Prince George , the eldest child of the Duke of Cambridge , followed by his sister , Princess Charlotte and younger brother , Prince Louis .
Sixth in line is Prince Henry of Wales , the younger son of the Prince of Wales . Any of the first six in line marrying without the sovereign 's consent would
be disqualified from succession . The first four individuals in the line of succession who are over 21 , and the sovereign 's consort , may be appointed
Counsellors of State . Counsellors of State perform some of the sovereign 's duties in the United Kingdom while he or she is out of the country or temporarily
incapacitated . Otherwise , individuals in the line of succession need not have specific legal or official roles . The United Kingdom is one of the 16
Commonwealth realms . Each of those countries has the same person as monarch and the same order of succession .
</item 6>

<item 7>
Succession to the British throne
Edward died childless in 1972 . Edward 's abdication was `` a demise of the Crown '' ( in the words of the Act ) , and the Duke of York , his brother who was
then next in the line , immediately succeeded to the throne and to its `` rights , privileges , and dignities '' , taking the regnal name George VI . He in turn
was succeeded in 1952 by his own elder daughter , Elizabeth II . By that time the monarch of the United Kingdom no longer reigned in the greater part of Ireland
( which had become a republic in 1949 ) , but was the monarch of a number of independent sovereign states ( Commonwealth realms ) . Commonwealth realms ( edit )
Further information : Perth Agreement Elizabeth II at her coronation in 1953 , passing to the left of the Coronation Chair Each of the Commonwealth realms has
the same person as monarch and , to maintain that arrangement , they have agreed to continue the same line of succession ; some realms do so through domestic
succession laws , while others stipulate whoever is monarch of the United Kingdom will also be monarch of that realm .
</item 7>

<item 8>
Succession to the British throne
3 : `` A person who ( when the person marries ) is one of the 6 persons next in the line of succession to the Crown must obtain the consent of Her Majesty
before marrying . '' Jump up ^ Most sources , including Whitaker 's Almanack , exclude her as a confirmed Catholic , but Debrett 's includes her in line . Jump
up ^ Albert and Leopold Windsor were listed on The Official Website of the British Monarchy until 2015 and in the 2013 edition of Whitaker 's Almanack as
following Estella Taylor ( b 2004 ) and eligible to succeed ; MSN News , Debrett 's and Whitaker 's Almanack 2015 lists them after Lady Amelia Windsor and
before Lady Helen Taylor . They were baptised as Catholics , and are not listed in line in editions of Whitaker 's earlier than 2012 . Jump up ^ Neither the
Legitimacy Act 1926 nor the Legitimacy Act 1959 changed the Law of Succession . Jump up ^ Per the His Majesty 's Declaration of Abdication Act 1936 .
</item 8>

<item 9>
Succession to the British throne
Each of those countries has the same person as monarch and the same order of succession . In 2011 , the prime ministers of the realms agreed unanimously to
adopt a common approach to amending the rules on the succession to their respective Crowns so that absolute primogeniture would apply for persons born after the
date of the agreement , instead of male - preference primogeniture , and the ban on marriages to Roman Catholics would be lifted , but the monarch would still
need to be in communion with the Church of England . After the necessary legislation had been enacted in accordance with each realm 's constitution , the
changes took effect on 26 March 2015 . Contents ( hide ) 1 Current line of succession 2 History 2.1 England 2.2 Scotland 2.3 After the union of the crowns 2.3.
1 Stuarts 2.3.
</item 9>

<item 10>
Succession to the British throne
In February 1952 , on her accession , Elizabeth II was proclaimed as sovereign separately throughout her realms . In October 2011 , the heads of government of
all 16 realms agreed unanimously at a meeting held in Perth , Australia , to proposed changes to the royal succession laws that would see the order of
succession for people born after 28 October 2011 governed by absolute primogeniture -- wherein succession passes to an individual 's children according to birth
order , regardless of gender -- instead of male - preference primogeniture . They also agreed to lift a ban on those who marry Roman Catholics . The ban on
Catholics themselves was retained to ensure that the monarch would be in communion with the Church of England . The changes came into effect across the
Commonwealth realms on 26 March 2015 , after legislation was made according to each realm 's constitution . Following the changes coming into effect , the
positions of the first 27 in line remained unchanged , including Princess Anne and her children and grandchildren , until the birth of Princess Charlotte of
Cambridge on 2 May 2015 .
</item 10>
=== END ITEM id=4f9dad478f40df78 ===

=== ITEM id=6879f6fcfb0a6be6 ===
Gold sentences and facts:
Sentence 0: The phrase `` objects in ( the ) mirror are closer than they appear '' is a safety warning that is required to be engraved on passenger side mirrors of motor vehicles in the USA , Canada , Nepal , India and Saudi Arabia . It is present because while these mirrors ' convexity gives them a useful field of view , it also makes objects appear smaller . Since smaller - appearing objects seem farther away than they actually are , a driver might make a maneuver such as a lane change assuming an adjacent vehicle is a safe distance behind , when in fact it is quite a bit closer . The warning serves as a reminder to the driver of this potential problem .
  Fact 0.0: The phrase 'objects in (the) mirror are closer than they appear' is a safety warning.
  Fact 0.1: The safety warning is required to be engraved on passenger side mirrors of motor vehicles.
  Fact 0.2: The engraving requirement applies in the USA, Canada, Nepal, India and Saudi Arabia.
  Fact 0.3: The warning is present on passenger side mirrors because their convexity gives them a useful field of view.
  Fact 0.4: The convexity of passenger side mirrors makes objects appear smaller.
  Fact 0.5: Smaller-appearing objects seem farther away than they actually are.
  Fact 0.6: A driver might make a maneuver such as a lane change assuming an adjacent vehicle is a safe distance behind.
  Fact 0.7: In fact, the adjacent vehicle is quite a bit closer than the driver assumes.
  Fact 0.8: The warning serves as a reminder to the driver of this potential problem.

Retrieved items:
<item 1>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 1>

<item 2>
Objects in mirror are closer than they appear
Iranian film Objects in Mirror ( fa ) See also ( edit ) Blind spot ( automobile ) Blind spot monitor Driver visibility Parking sensors Rear - view mirror Mind
the gap , another safety warning used at various London Underground stations that has also become a well - known catchphrase . References ( edit ) Jump up ^ For
example , in the U.S , PART 571 Federal Motor Vehicle Safety Standards , Section 571.111 S5. 4.2 `` Each convex mirror shall have permanently and indelibly
marked at the lower edge of the mirror 's reflective surface , in letters not less than 4.8 mm nor more than 6.4 mm high the words `` Objects in Mirror Are
Closer Than They Appear . '' Jump up ^ Why does the passenger side window on my car state ' objects in mirror are closer than they appear ? ' Explanation from
PhysLink.com .
</item 2>

<item 3>
Vanishing point
If we know the coordinates of one of these points , say v , and the direction of a straight line on the image plane , which passes through a second point , say
v , we can compute the coordinates of both v and v 3 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x ,
y , f ) , v ≡ ( x , y , f ) be the three corresponding vanishing points respectively . The orthocenter of the triangle with vertices in the three vanishing
points is the intersection of the optical axis and the image plane . Curvilinear and reverse perspective ( edit ) A curvilinear perspective is a drawing with
either 4 or 5 vanishing points . In 5 - point perspective the vanishing points are mapped into a circle with 4 vanishing points at the cardinal headings N , W ,
S , E and one at the circle 's origin . A reverse perspective is a drawing with vanishing points that are placed outside the painting with the illusion that
they are `` in front of '' the painting . Single point perspective projection . Double point perspective projection .
</item 3>

<item 4>
Objects in mirror are closer than they appear
In popular culture ( edit ) Despite its origin as a utilitarian safety warning , the phrase has become a well known catch phrase that has been used for many
other purposes . Some of them are : Objects in the Mirror Are Closer Than They Appear , the 2000 debut album of rap rock group Confrontation Camp . Objects in
the Mirror Are Closer Than They Appear , a 1993 album by Nancy Moran Objects in Mirror Are Closer Than They Appear , a 2005 album by Jozef van Wissem . Objects
in Mirror Are Closer Than They Appear , a 1995 novel by Katharine Weber . Showing an approach of a Tyrannosaurus rex in the film Jurassic Park , which was
parodied in Toy Story 2 . A Far Side cartoon by Gary Larson in which the side mirror showed a very large eyeball . A reference in Ace Ventura : Pet Detective ,
in which a pejorative term replaces `` objects . '' `` Objects in the Rear View Mirror May Appear Closer than They Are , '' a 1994 song performed by Meat Loaf
and written by Jim Steinman - the opposite of the safety warning .
</item 4>

<item 5>
Drawing
Typically this convergence point is somewhere along the horizon , as buildings are built level with the flat surface . When multiple structures are aligned with
each other , such as buildings along a street , the horizontal tops and bottoms of the structures typically converge at a vanishing point . When both the fronts
and sides of a building are drawn , then the parallel lines forming a side converge at a second point along the horizon ( which may be off the drawing paper . )
This is a two - point perspective . Converging the vertical lines to a third point above or below the horizon then produces a three - point perspective . Depth
can also be portrayed by several techniques in addition to the perspective approach above . Objects of similar size should appear ever smaller the further they
are from the viewer . Thus the back wheel of a cart appears slightly smaller than the front wheel . Depth can be portrayed through the use of texture . As the
texture of an object gets further away it becomes more compressed and busy , taking on an entirely different character than if it was close . Depth can also be
portrayed by reducing the contrast in more distant objects , and by making their colors less saturated .
</item 5>

<item 6>
Vanishing point
A , B , C , D and V are points on the image , their separation given in pixels ; A ' , B ' , C ' and D ' are in the real world , their separation in metres . In
( 1 ) , the width of the side street , W is computed from the known widths of the adjacent shops . In ( 2 ) , the width of only one shop is needed because a
vanishing point , V is visible . Camera calibration : The vanishing points of an image contain important information for camera calibration . Various
calibration techniques have been introduced using the properties of vanishing points to find intrinsic and extrinsic calibration parameters . 3D reconstruction
: A man - made environment has two main characteristics -- several lines in the scene are parallel , and a number of edges present are orthogonal . Vanishing
points aid in comprehending the environment . Using sets of parallel lines in the plane , the orientation of the plane can be calculated using vanishing points
. Torre and Coelho performed extensive investigation in the use of vanishing points to implement a full system .
</item 6>

<item 7>
Objects in mirror are closer than they appear
Explanation from PhysLink.com . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Objects_in_mirror_are_closer_than_they_appear&oldid=838121569 ''
Categories : English phrases Mirrors Car safety Hidden categories : Wikipedia articles needing clarification from March 2016 Interlanguage link template link
number Talk About Wikipedia 한국어 Edit links This page was last edited on 25 April 2018 , at 01 : 58 . Text is available under the Creative Commons Attribution -
ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark
of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Objects in mirror are closer than they appear
A reference in the 2011 film Transformers : Dark of the Moon film occurs when Dino , Bumblebee and Sideswipe are protecting Sentinel from the dreads , this
appears in a brief shot in Dino 's mirror . `` Objects In The Mirror , '' a song performed by American rapper Mac Miller on his second studio album Watching
Movies with the Sound Off `` Subjects in the mirror are more far than they appear '' the title of experimental visual project by Hungarian director / production
designer Pater Sparrow `` Objects in the Mirror , '' a 2015 song by a-ha . Objects in the mirror , a song performed by Chris Miles . ( 2016 ) Objects in mirror
are closer than they appear was a level in the videogame `` Bugs Bunny : Lost in time '' ( PlayStation 1 ) . In My Little Pony : The Movie , the sentence ``
Objects in windows may be less mountainous than they appear '' is uttered .
</item 8>

<item 9>
Objects in mirror are closer than they appear
Objects in mirror are Closer than they appear - wikipedia Objects in mirror are Closer than they appear Jump to : navigation , search Side - view mirror with
legend Wing mirror on Korean - specification vehicle . Legend in Korean reads `` Objects in mirror are closer than they appear '' . Main article : Wing mirror
The phrase `` objects in ( the ) mirror are closer than they appear '' is a safety warning that is required to be engraved on passenger side mirrors of motor
vehicles in the USA , Canada , Nepal , India and Saudi Arabia . It is present because while these mirrors ' convexity gives them a useful field of view , it
also makes objects appear smaller . Since smaller - appearing objects seem farther away than they actually are , a driver might make a maneuver such as a lane
change assuming an adjacent vehicle is a safe distance behind , when in fact it is quite a bit closer . The warning serves as a reminder to the driver of this
potential problem . Comparison of the real over-the - shoulder view ( green arrow ) and apparent through the passenger side mirror ( sum of red arrows )
evaluation of distance between the driver 's head and the object behind .
</item 9>

<item 10>
Vanishing point
With the assumption that the environment consists of objects with only parallel or perpendicular sides , also called Lego - land , using vanishing points
constructed in a single image of the scene they recovered the 3D geometry of the scene . Similar ideas are also used in the field of robotics , mainly in
navigation and autonomous vehicles , and in areas concerned with object detection . See also ( edit ) Graphical projection References ( edit ) ^ Jump up to :
Kirsti Andersen ( 2007 ) Geometry of an Art , p. xxx , Springer , ISBN 0 - 387 - 25961 - 9 ^ Jump up to : B. Caprile , V. Torre ( 1 ) `` Using Vanishing Points
for Camera Calibration '' , International Journal of Computer Vision , Volume 4 , Issue 2 , pp. 127 - 139 , March 1990 Jump up ^ H. Ditton ( 1712 ) Treatise on
Perspective , p 45 Jump up ^ S.T. Barnard ' Interpreting Perspective Images `` , Artificial Intelligence 21 , 1983 , pp.
</item 10>
=== END ITEM id=6879f6fcfb0a6be6 ===

=== ITEM id=54671dae363904dc ===
Gold sentences and facts:
Sentence 0: The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of the Mahanadi Delta .
  Fact 0.0: The Eastern Coastal Plains refers to a wide stretch of landmass of India.
  Fact 0.1: The Eastern Coastal Plains lie between the Eastern Ghats and the Bay of Bengal.
  Fact 0.2: The Eastern Coastal Plains are wider and more level than the western coastal plains.
  Fact 0.3: The Eastern Coastal Plains stretch from Tamil Nadu in the south to West Bengal in the north.
  Fact 0.4: The Eastern Coastal Plains run through Andhra Pradesh and Odisha.
  Fact 0.5: Chilka Lake is a brackish water lake.
  Fact 0.6: Chilka Lake lies along the eastern coastal plain.
  Fact 0.7: Chilka Lake lies in the state of Odisha.
  Fact 0.8: Chilka Lake stretches to the south of the Mahanadi Delta.

Retrieved items:
<item 1>
Sub-Saharan Africa
Arabian Peninsula Najd Hejaz Tihamah Eastern Arabia South Arabia Hadhramaut Arabian Peninsula coastal fog desert Tigris -- Euphrates Mesopotamia Upper
Mesopotamia Lower Mesopotamia Sawad Nineveh plains Akkad ( region ) Babylonia Canaan Aram Eber - Nari Suhum Eastern Mediterranean Mashriq Kurdistan Levant
Southern Levant Transjordan Jordan Rift Valley Israel Levantine Sea Golan Heights Hula Valley Galilee Gilead Judea Samaria Arabah Anti-Lebanon Mountains Sinai
Peninsula Arabian Desert Syrian Desert Fertile Crescent Azerbaijan Syria Palestine Iranian Plateau Armenian Highlands Caucasus Caucasus Mountains Greater
Caucasus Lesser Caucasus North Caucasus South Caucasus Kur - Araz Lowland Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt
South Greater India Indian subcontinent Himalayas Hindu Kush Western Ghats Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir
Kashmir Valley Pir Panjal Range Thar Desert Indus Valley Indus River Delta Indus Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal
Plains Meghalaya subtropical forests MENASA Lower
</item 1>

<item 2>
Eastern coastal plains
pp. 20 -- 21 . ISBN 8171725163 . Geography of India Climate Climate Climatic regions Geology Fossil Parks Geology of India Indian Plate Stones Landforms Beaches
Canals Desert Extreme points Glaciers Islands Lakes Mountains Rivers Valleys Volcanoes Waterfalls Plains Indo - Gangetic Eastern Coastal Western Coastal Regions
North India Northeast India East India South India West India Central India Subdivisions Autonomous administrative divisions States and territories Districts
Cities Towns Municipalities Environment Biogeographic classification Ecoregions Fauna Flora Forests Issues Protected Areas Wildlife This Indian location article
is a stub . You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Eastern_coastal_plains&oldid=812520937 ''
Categories : Plains of India Coasts of India India geography stubs Hidden categories : Articles needing additional references from May 2015 All articles needing
additional references India articles missing geocoordinate data All articles needing coordinates All stub articles Talk About Wikipedia हिन्दी Português தமிழ்
Edit links This page was last edited on 28 November 2017 , at 08 : 32 .
</item 2>

<item 3>
Eastern coastal plains
Eastern coastal plains - Wikipedia Eastern coastal plains Jump to : navigation , search This article relies largely or entirely on a single source . Relevant
discussion may be found on the talk page . Please help improve this article by introducing citations to additional sources . ( May 2015 ) View of Fields at
Biccavolu , Eastern coastal plains , Andhra Pradesh The Eastern Coastal Plains refers to a wide stretch of landmass of India , lying between the Eastern Ghats
and the Bay of Bengal . It is wider and leveled than the western coastal plains and stretches from Tamil Nadu in the south to West Bengal in the north through
Andhra Pradesh and Odisha . Chilka Lake is a brackish water lake along the eastern coastal plain . It lies in the state of Odisha and stretches to the south of
the Mahanadi Delta . Deltas of many of India 's rivers form a major portion of these plains . The Mahanadi , Godavari , Kaveri and Krishna rivers drain these
plains . The region receives both the Northeast & Southwest monsoon rains with its annual rainfall averaging between 1,000 and 3,000 mm ( 39 and 118 in ) .
</item 3>

<item 4>
Graph query row:
entity: Eastern Coastal Plains
relationship: LOCATED_IN
related: West Bengal
fact: None
context: None
relation: None
</item 4>

<item 5>
Graph query row:
entity: Eastern Coastal Plains
relationship: LOCATED_IN
related: India
fact: None
context: None
relation: None
</item 5>

<item 6>
Dead Sea
Kur - Araz Lowland Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt South Greater India Indian subcontinent Himalayas Hindu
Kush Western Ghats Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir Kashmir Valley Pir Panjal Range Thar Desert Indus Valley
Indus River Delta Indus Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic
plains moist deciduous forests Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau
Coromandel Coast Konkan False Divi Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen Glacier Bay
of Bengal Gulf of Khambhat Gulf of Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands Andaman
Islands Nicobar Islands Maldive Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater
</item 6>

<item 7>
Eastern coastal plains
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Ring of Fire
Aram Eber - Nari Suhum Eastern Mediterranean Mashriq Kurdistan Levant Southern Levant Transjordan Jordan Rift Valley Levantine Sea Golan Heights Hula Valley
Gaza Strip West Bank Galilee Gilead Judea Samaria Arabah Anti-Lebanon Mountains Sinai Peninsula Arabian Desert Syrian Desert Fertile Crescent Azerbaijan Syria
Palestine Iranian Plateau Armenian Highlands Caucasus Caucasus Mountains Greater Caucasus Lesser Caucasus North Caucasus South Caucasus Kur - Araz Lowland
Lankaran Lowland Alborz Absheron Peninsula Anatolia Cilicia Cappadocia Alpide belt South Greater India Indian subcontinent Himalayas Hindu Kush Western Ghats
Eastern Ghats Ganges Basin Ganges Delta Pashtunistan Punjab Balochistan Kashmir Kashmir Valley Pir Panjal Range Thar Desert Indus Valley Indus River Delta Indus
Valley Desert Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic plains moist deciduous
forests Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau Coromandel Coast Konkan False
Divi Point Hindi Belt Ladakh Aksai Chin
</item 8>

<item 9>
Sub-Saharan Africa
Indo - Gangetic Plain Eastern coastal plains Western Coastal Plains Meghalaya subtropical forests MENASA Lower Gangetic plains moist deciduous forests
Northwestern Himalayan alpine shrub and meadows Doab Bagar tract Great Rann of Kutch Little Rann of Kutch Deccan Plateau Coromandel Coast Konkan False Divi
Point Hindi Belt Ladakh Aksai Chin Gilgit - Baltistan Baltistan Shigar Valley Karakoram Saltoro Mountains Siachen Glacier Bay of Bengal Gulf of Khambhat Gulf of
Kutch Gulf of Mannar Trans - Karakoram Tract Wakhan Corridor Wakhjir Pass Lakshadweep Andaman and Nicobar Islands Andaman Islands Nicobar Islands Maldive
Islands Alpide belt Southeast Mainland Indochina Malay Peninsula Maritime Peninsular Malaysia Sunda Islands Greater Sunda Islands Lesser Sunda Islands
Indonesian Archipelago Timor New Guinea Bonis Peninsula Papuan Peninsula Huon Peninsula Huon Gulf Bird 's Head Peninsula Gazelle Peninsula Philippine
Archipelago Luzon Visayas Mindanao Leyte Gulf Gulf of Thailand East Indies Nanyang Alpide belt Asia - Pacific Tropical Asia Ring of Fire Regions of Europe North
Nordic Northwestern Scandinavia Scandinavian Peninsula
</item 9>

<item 10>
Eastern coastal plains
The width of the plains varies between 100 and 130 km ( 62 to 80 miles ) . It is locally known as Utkal Plains in the Northern part between Cossye and
Rushikulya Rivers , Northern Circars in the Central part between Rushikulya and Krishna Rivers and , as Coromandel Coast in the Southern part from the south of
river Krishna till the Southern tip of Mainland India at Cape Comorin where it merges with the Western Coastal Plains . See also ( edit ) Western Coastal Plains
References ( edit ) Jump up ^ Sr. Bimcy ; Sr. Sisily ; Charlotte . Bibliographic information . Scholar Publishing House . pp. 20 -- 21 . ISBN 8171725163 .
</item 10>
=== END ITEM id=54671dae363904dc ===

=== ITEM id=d63fa2b424080524 ===
Gold sentences and facts:
Sentence 0: NCAA Men 's National Collegiate Volleyball Championship Sport College indoor volleyball Founded 1970 No. of teams 6 ( as of 2014 ) Country United States Most recent champion ( s ) Ohio State 2 ( 3 ) TV partner ( s ) ESPNU Official website NCAA.com
  Fact 0.0: The NCAA Men's National Collegiate Volleyball Championship is a college indoor volleyball event.
  Fact 0.1: The NCAA Men's National Collegiate Volleyball Championship was founded in 1970.
  Fact 0.2: The NCAA Men's National Collegiate Volleyball Championship had 6 teams as of 2014.
  Fact 0.3: The NCAA Men's National Collegiate Volleyball Championship is held in the United States.
  Fact 0.4: Ohio State is listed as the most recent champion, with a count of 2 (3).
  Fact 0.5: The TV partner of the NCAA Men's National Collegiate Volleyball Championship is ESPNU.
  Fact 0.6: The official website of the NCAA Men's National Collegiate Volleyball Championship is NCAA.com.
Sentence 1: NCAA Men 's National Collegiate Volleyball Championship Year Site ( Host ) Host Arena Final Third Place Final / Other participants Winner Score Runner - up Third Place Score Fourth Place 1970 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 1 ) 3 -- 0 Long Beach State UC Santa Barbara 2 -- 0 Ball State 1971 Details UCLA ( 29 -- 1 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 0 Springfield 1972 Details Muncie , Indiana ( Ball State ) Irving Gymnasium UCLA ( 27 -- 7 ) 3 -- 2 San Diego State Ball State 2 -- 0 UC Santa Barbara 1973 Details San Diego ( San Diego State ) Peterson Gym San Diego State ( 21 -- 5 ) 3 -- 1 Long Beach State Ball State 2 -- 0 Army Details Santa Barbara , California ( UCSB ) Robertson Gymnasium UCLA ( 30 -- 5 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 1 Springfield Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 27 -- 8 ) 3 -- 1 UC Santa Barbara Ohio State 2 -- 0 Yale 1976 Details Muncie , Indiana ( Ball State ) Irving Gymnasium UCLA ( 15 -- 2 ) 3 -- 0 Pepperdine Ohio State 2 -- 0 Springfield 1977 Details Los Angeles ( UCLA ) Pauley Pavilion USC ( 18 -- 1 ) 3 -- 1 Ohio State Pepperdine 2 -- 0 Rutgers -- Newark 1978 Details Columbus , Ohio ( Ohio State ) St. John Arena Pepperdine ( 21 -- 4 ) 3 -- 2 UCLA Ohio State 2 -- 0 Rutgers -- Newark 1979 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 30 -- 0 ) 3 -- 1 USC Rutgers -- Newark 3 -- 2 Ball State 1980 Details Muncie , Indiana ( Ball State ) Irving Gymnasium USC ( 22 -- 6 ) 3 -- 1 UCLA Ohio State 3 -- 0 Rutgers -- Newark 1981 Details Santa Barbara , California ( UCSB ) UCSB Events Center UCLA ( 32 -- 3 ) 3 -- 2 USC Penn State 3 -- 1 Ohio State 1982 Details University Park , Pennsylvania ( Penn State ) Rec Hall UCLA ( 29 -- 0 ) 3 -- 0 Penn State USC 2 -- 1 Ohio State Details Columbus , Ohio ( Ohio State ) St. John Arena UCLA ( 27 -- 4 ) 3 -- 0 Pepperdine Ohio State 3 -- 1 Penn State 1984 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 38 -- 0 ) 3 -- 1 Pepperdine George Mason 3 -- 0 Ball State 1985 Details Pepperdine ( 25 -- 2 ) 3 -- 2 USC George Mason 3 -- 0 Ball State 1986 Details University Park , Pennsylvania ( Penn State ) Rec Hall Pepperdine ( 22 -- 7 ) 3 -- 2 USC Penn State 3 -- 0 Ohio State Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 28 -- 3 ) 3 -- 0 USC Penn State 3 -- 0 Ohio State 1988 Details Fort Wayne , Indiana ( IPFW ) ACWMC USC ( 34 -- 4 ) 3 -- 2 UC Santa Barbara Ball State 3 -- 1 George Mason 1989 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 29 -- 5 ) 3 -- 1 Stanford Penn State 3 -- 0 Ball State 1990 Details Fairfax , Virginia ( George Mason ) Patriot Center USC ( 26 -- 7 ) 3 -- 1 Long Beach State Ball State 3 -- 1 Rutgers -- Newark 1991 Details Honolulu , HI ( Hawaii ) Neal S. Blaisdell Center Long Beach State ( 31 -- 4 ) 3 -- 1 USC IPFW 3 -- 1 Penn State 1992 Details Muncie , Indiana ( Ball State ) John E. Worthen Arena Pepperdine ( 24 -- 4 ) 3 -- 0 Stanford Penn State 3 -- 0 IPFW 1993 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 3 ) 3 -- 0 Cal State Northridge Penn State 3 -- 2 Ohio State 1994 Details Fort Wayne , Indiana ( IPFW ) ACWMC Penn State ( 26 -- 3 ) 3 -- 2 UCLA Ball State 3 -- 0 IPFW 1995 Details Springfield , Massachusetts ( Springfield ) Springfield Civic Center UCLA ( 31 -- 1 ) 3 -- 0 Penn State Ball State 3 -- 1 Hawaii Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 26 -- 5 ) 3 -- 2 Hawaii Lewis and Penn State 1997 Details Columbus , Ohio ( Ohio State ) St. John Arena Stanford ( 27 -- 4 ) 3 -- 2 UCLA Ball State and Penn State 1998 Details Honolulu , HI ( Hawaii ) Stan Sheriff Center UCLA ( 28 -- 4 ) 3 -- 0 Pepperdine Lewis and Princeton 1999 Details Los Angeles ( UCLA ) Pauley Pavilion BYU ( 30 -- 1 ) 3 -- 0 Long Beach State IPFW and Penn State 2000 Details Fort Wayne , Indiana ( IPFW ) ACWMC UCLA ( 29 -- 5 ) 3 -- 0 Ohio State Penn State and Pepperdine 2001 Details Long Beach , California ( Long Beach State ) The Pyramid BYU ( 23 -- 4 ) 3 -- 0 UCLA Ohio State and Penn State 2002 Details University Park , Pennsylvania ( Penn State ) Rec Hall Hawaii ( 24 -- 8 ) † 3 -- 1 Pepperdine Ball State and Penn State 2003 Details Long Beach , California ( Long Beach State ) The Pyramid Lewis ( 29 -- 6 ) † 3 -- 2 BYU Penn State and Pepperdine Details Honolulu , HI ( Hawaii ) Stan Sheriff Center BYU ( 29 -- 4 ) 3 -- 2 Long Beach State Lewis and Penn State 2005 Details Los Angeles ( UCLA ) Pauley Pavilion Pepperdine ( 25 -- 2 ) 3 -- 2 UCLA Ohio State and Penn State 2006 Details University Park , Pennsylvania ( Penn State ) Rec Hall UCLA ( 26 -- 12 ) 3 -- 0 Penn State UC Irvine and IPFW 2007 Details Columbus , Ohio ( Ohio State ) St. John Arena UC Irvine ( 29 -- 5 ) 3 -- 1 IPFW Penn State and Pepperdine 2008 Details Irvine , California ( UC Irvine ) Bren Events Center Penn State ( 30 -- 1 ) 3 -- 1 Pepperdine Long Beach State and Ohio State 2009 Details Provo , Utah ( BYU ) Smith Fieldhouse UC Irvine ( 27 -- 5 ) 3 -- 2 USC ( 21 -- 11 ) Ohio State and Penn State Details Stanford , California ( Stanford ) Maples Pavilion Stanford ( 24 -- 6 ) 3 -- 0 Penn State ( 24 -- 8 ) Cal State Northridge and Ohio State 2011 Details University Park , Pennsylvania ( Penn State ) Rec Hall Ohio State ( 26 -- 6 ) 3 -- 2 UC Santa Barbara ( 18 -- 15 ) Penn State and USC 2012 Details Los Angeles ( USC ) Galen Center UC Irvine ( 26 -- 5 ) 3 -- 0 USC ( 24 -- 6 ) Lewis and Penn State 2013 Details Los Angeles ( UCLA ) Pauley Pavilion UC Irvine ( 25 -- 7 ) 3 -- 0 BYU ( 26 -- 5 ) Loyola Chicago and Penn State 2014 Details Chicago ( Loyola Chicago ) Gentile Arena Loyola Chicago ( 29 - 1 ) 3 - 1 Stanford ( 24 - 9 ) 3rd -- BYU and Penn State 5th -- Lewis and Erskine 2015 Details Stanford , California ( Stanford ) Maples Pavilion Loyola Chicago ( 28 - 2 ) 3 - 2 Lewis ( 27 - 4 ) 3rd -- UC Irvine and Penn State 5th -- Hawaii and Pfeiffer 2016 Details University Park , Pennsylvania ( Penn State ) Rec Hall Ohio State ( 31 - 2 ) 3 - 0 BYU ( 27 - 4 ) 3rd -- UCLA and Long Beach State 5th -- Erskine and George Mason 2017 Details Columbus , Ohio ( Ohio State ) St. John Arena Ohio State ( 32 -- 2 ) 3 - 0 BYU ( 26 - 5 ) 3rd -- Hawaii and Long Beach State 5th -- Barton and Penn State 2018 Details Los Angeles ( UCLA ) Pauley Pavilion 2019 Details Long Beach , California ( Long Beach State ) The Pyramid 2020 Details Fairfax , Virginia ( George Mason ) EagleBank Arena 2021 Details Columbus , Ohio ( Ohio State ) Covelli Arena 2022 Details Los Angeles ( UCLA ) Pauley Pavilion
  Fact 1.0: In 1970, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.1: In 1970, UCLA (24-1) won the final 3-0 over runner-up Long Beach State.
  Fact 1.2: In 1970, third place went to UC Santa Barbara, with a third-place score of 2-0.
  Fact 1.3: In 1970, fourth place went to Ball State.
  Fact 1.4: In 1971, UCLA (29-1) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.5: In 1971, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.6: In 1971, fourth place went to Springfield.
  Fact 1.7: In 1972, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.8: In 1972, UCLA (27-7) won the final 3-2 over runner-up San Diego State.
  Fact 1.9: In 1972, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.10: In 1972, fourth place went to UC Santa Barbara.
  Fact 1.11: In 1973, the championship was held in San Diego (host San Diego State) at Peterson Gym.
  Fact 1.12: In 1973, San Diego State (21-5) won the final 3-1 over runner-up Long Beach State.
  Fact 1.13: In 1973, third place went to Ball State, with a third-place score of 2-0.
  Fact 1.14: In 1973, fourth place went to Army.
  Fact 1.15: In an unlabeled row after 1973, the championship was held in Santa Barbara, California (host UCSB) at Robertson Gymnasium.
  Fact 1.16: In that unlabeled row, UCLA (30-5) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.17: In that unlabeled row, third place went to Ball State, with a third-place score of 2-1.
  Fact 1.18: In that unlabeled row, fourth place went to Springfield.
  Fact 1.19: In an unlabeled row after the Santa Barbara row, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.20: In that unlabeled row, UCLA (27-8) won the final 3-1 over runner-up UC Santa Barbara.
  Fact 1.21: In that unlabeled row, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.22: In that unlabeled row, fourth place went to Yale.
  Fact 1.23: In 1976, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.24: In 1976, UCLA (15-2) won the final 3-0 over runner-up Pepperdine.
  Fact 1.25: In 1976, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.26: In 1976, fourth place went to Springfield.
  Fact 1.27: In 1977, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.28: In 1977, USC (18-1) won the final 3-1 over runner-up Ohio State.
  Fact 1.29: In 1977, third place went to Pepperdine, with a third-place score of 2-0.
  Fact 1.30: In 1977, fourth place went to Rutgers-Newark.
  Fact 1.31: In 1978, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.32: In 1978, Pepperdine (21-4) won the final 3-2 over runner-up UCLA.
  Fact 1.33: In 1978, third place went to Ohio State, with a third-place score of 2-0.
  Fact 1.34: In 1978, fourth place went to Rutgers-Newark.
  Fact 1.35: In 1979, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.36: In 1979, UCLA (30-0) won the final 3-1 over runner-up USC.
  Fact 1.37: In 1979, third place went to Rutgers-Newark, with a third-place score of 3-2.
  Fact 1.38: In 1979, fourth place went to Ball State.
  Fact 1.39: In 1980, the championship was held in Muncie, Indiana (host Ball State) at Irving Gymnasium.
  Fact 1.40: In 1980, USC (22-6) won the final 3-1 over runner-up UCLA.
  Fact 1.41: In 1980, third place went to Ohio State, with a third-place score of 3-0.
  Fact 1.42: In 1980, fourth place went to Rutgers-Newark.
  Fact 1.43: In 1981, the championship was held in Santa Barbara, California (host UCSB) at the UCSB Events Center.
  Fact 1.44: In 1981, UCLA (32-3) won the final 3-2 over runner-up USC.
  Fact 1.45: In 1981, third place went to Penn State, with a third-place score of 3-1.
  Fact 1.46: In 1981, fourth place went to Ohio State.
  Fact 1.47: In 1982, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.48: In 1982, UCLA (29-0) won the final 3-0 over runner-up Penn State.
  Fact 1.49: In 1982, third place went to USC, with a third-place score of 2-1.
  Fact 1.50: In 1982, fourth place went to Ohio State.
  Fact 1.51: In an unlabeled row, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.52: In that unlabeled row, UCLA (27-4) won the final 3-0 over runner-up Pepperdine.
  Fact 1.53: In that unlabeled row, third place went to Ohio State, with a third-place score of 3-1.
  Fact 1.54: In that unlabeled row, fourth place went to Penn State.
  Fact 1.55: In 1984, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.56: In 1984, UCLA (38-0) won the final 3-1 over runner-up Pepperdine.
  Fact 1.57: In 1984, third place went to George Mason, with a third-place score of 3-0.
  Fact 1.58: In 1984, fourth place went to Ball State.
  Fact 1.59: In 1985, Pepperdine (25-2) won the final 3-2 over runner-up USC.
  Fact 1.60: In 1985, third place went to George Mason, with a third-place score of 3-0.
  Fact 1.61: In 1985, fourth place went to Ball State.
  Fact 1.62: In 1986, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.63: In 1986, Pepperdine (22-7) won the final 3-2 over runner-up USC.
  Fact 1.64: In 1986, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.65: In 1986, fourth place went to Ohio State.
  Fact 1.66: In an unlabeled row, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.67: In that unlabeled row, UCLA (28-3) won the final 3-0 over runner-up USC.
  Fact 1.68: In that unlabeled row, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.69: In that unlabeled row, fourth place went to Ohio State.
  Fact 1.70: In 1988, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.71: In 1988, USC (34-4) won the final 3-2 over runner-up UC Santa Barbara.
  Fact 1.72: In 1988, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.73: In 1988, fourth place went to George Mason.
  Fact 1.74: In 1989, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.75: In 1989, UCLA (29-5) won the final 3-1 over runner-up Stanford.
  Fact 1.76: In 1989, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.77: In 1989, fourth place went to Ball State.
  Fact 1.78: In 1990, the championship was held in Fairfax, Virginia (host George Mason) at Patriot Center.
  Fact 1.79: In 1990, USC (26-7) won the final 3-1 over runner-up Long Beach State.
  Fact 1.80: In 1990, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.81: In 1990, fourth place went to Rutgers-Newark.
  Fact 1.82: In 1991, the championship was held in Honolulu, HI (host Hawaii) at Neal S. Blaisdell Center.
  Fact 1.83: In 1991, Long Beach State (31-4) won the final 3-1 over runner-up USC.
  Fact 1.84: In 1991, third place went to IPFW, with a third-place score of 3-1.
  Fact 1.85: In 1991, fourth place went to Penn State.
  Fact 1.86: In 1992, the championship was held in Muncie, Indiana (host Ball State) at John E. Worthen Arena.
  Fact 1.87: In 1992, Pepperdine (24-4) won the final 3-0 over runner-up Stanford.
  Fact 1.88: In 1992, third place went to Penn State, with a third-place score of 3-0.
  Fact 1.89: In 1992, fourth place went to IPFW.
  Fact 1.90: In 1993, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.91: In 1993, UCLA (24-3) won the final 3-0 over runner-up Cal State Northridge.
  Fact 1.92: In 1993, third place went to Penn State, with a third-place score of 3-2.
  Fact 1.93: In 1993, fourth place went to Ohio State.
  Fact 1.94: In 1994, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.95: In 1994, Penn State (26-3) won the final 3-2 over runner-up UCLA.
  Fact 1.96: In 1994, third place went to Ball State, with a third-place score of 3-0.
  Fact 1.97: In 1994, fourth place went to IPFW.
  Fact 1.98: In 1995, the championship was held in Springfield, Massachusetts (host Springfield) at Springfield Civic Center.
  Fact 1.99: In 1995, UCLA (31-1) won the final 3-0 over runner-up Penn State.
  Fact 1.100: In 1995, third place went to Ball State, with a third-place score of 3-1.
  Fact 1.101: In 1995, fourth place went to Hawaii.
  Fact 1.102: In an unlabeled row between 1995 and 1997, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.103: In that unlabeled row, UCLA (26-5) won the final 3-2 over runner-up Hawaii.
  Fact 1.104: In that unlabeled row, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.105: In 1997, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.106: In 1997, Stanford (27-4) won the final 3-2 over runner-up UCLA.
  Fact 1.107: In 1997, Ball State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.108: In 1998, the championship was held in Honolulu, HI (host Hawaii) at Stan Sheriff Center.
  Fact 1.109: In 1998, UCLA (28-4) won the final 3-0 over runner-up Pepperdine.
  Fact 1.110: In 1998, Lewis and Princeton are listed as the remaining finishers, with no score.
  Fact 1.111: In 1999, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.112: In 1999, BYU (30-1) won the final 3-0 over runner-up Long Beach State.
  Fact 1.113: In 1999, IPFW and Penn State are listed as the remaining finishers, with no score.
  Fact 1.114: In 2000, the championship was held in Fort Wayne, Indiana (host IPFW) at ACWMC.
  Fact 1.115: In 2000, UCLA (29-5) won the final 3-0 over runner-up Ohio State.
  Fact 1.116: In 2000, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.117: In 2001, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.118: In 2001, BYU (23-4) won the final 3-0 over runner-up UCLA.
  Fact 1.119: In 2001, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.120: In 2002, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.121: In 2002, Hawaii (24-8), marked with a dagger, won the final 3-1 over runner-up Pepperdine.
  Fact 1.122: In 2002, Ball State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.123: In 2003, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.124: In 2003, Lewis (29-6), marked with a dagger, won the final 3-2 over runner-up BYU.
  Fact 1.125: In 2003, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.126: In an unlabeled row between 2003 and 2005, the championship was held in Honolulu, HI (host Hawaii) at Stan Sheriff Center.
  Fact 1.127: In that unlabeled row, BYU (29-4) won the final 3-2 over runner-up Long Beach State.
  Fact 1.128: In that unlabeled row, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.129: In 2005, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.130: In 2005, Pepperdine (25-2) won the final 3-2 over runner-up UCLA.
  Fact 1.131: In 2005, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.132: In 2006, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.133: In 2006, UCLA (26-12) won the final 3-0 over runner-up Penn State.
  Fact 1.134: In 2006, UC Irvine and IPFW are listed as the remaining finishers, with no score.
  Fact 1.135: In 2007, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.136: In 2007, UC Irvine (29-5) won the final 3-1 over runner-up IPFW.
  Fact 1.137: In 2007, Penn State and Pepperdine are listed as the remaining finishers, with no score.
  Fact 1.138: In 2008, the championship was held in Irvine, California (host UC Irvine) at Bren Events Center.
  Fact 1.139: In 2008, Penn State (30-1) won the final 3-1 over runner-up Pepperdine.
  Fact 1.140: In 2008, Long Beach State and Ohio State are listed as the remaining finishers, with no score.
  Fact 1.141: In 2009, the championship was held in Provo, Utah (host BYU) at Smith Fieldhouse.
  Fact 1.142: In 2009, UC Irvine (27-5) won the final 3-2 over runner-up USC (21-11).
  Fact 1.143: In 2009, Ohio State and Penn State are listed as the remaining finishers, with no score.
  Fact 1.144: In an unlabeled row between 2009 and 2011, the championship was held in Stanford, California (host Stanford) at Maples Pavilion.
  Fact 1.145: In that unlabeled row, Stanford (24-6) won the final 3-0 over runner-up Penn State (24-8).
  Fact 1.146: In that unlabeled row, Cal State Northridge and Ohio State are listed as the remaining finishers, with no score.
  Fact 1.147: In 2011, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.148: In 2011, Ohio State (26-6) won the final 3-2 over runner-up UC Santa Barbara (18-15).
  Fact 1.149: In 2011, Penn State and USC are listed as the remaining finishers, with no score.
  Fact 1.150: In 2012, the championship was held in Los Angeles (host USC) at Galen Center.
  Fact 1.151: In 2012, UC Irvine (26-5) won the final 3-0 over runner-up USC (24-6).
  Fact 1.152: In 2012, Lewis and Penn State are listed as the remaining finishers, with no score.
  Fact 1.153: In 2013, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.154: In 2013, UC Irvine (25-7) won the final 3-0 over runner-up BYU (26-5).
  Fact 1.155: In 2013, Loyola Chicago and Penn State are listed as the remaining finishers, with no score.
  Fact 1.156: In 2014, the championship was held in Chicago (host Loyola Chicago) at Gentile Arena.
  Fact 1.157: In 2014, Loyola Chicago (29-1) won the final 3-1 over runner-up Stanford (24-9).
  Fact 1.158: In 2014, BYU and Penn State are listed as third place.
  Fact 1.159: In 2014, Lewis and Erskine are listed as fifth place.
  Fact 1.160: In 2015, the championship was held in Stanford, California (host Stanford) at Maples Pavilion.
  Fact 1.161: In 2015, Loyola Chicago (28-2) won the final 3-2 over runner-up Lewis (27-4).
  Fact 1.162: In 2015, UC Irvine and Penn State are listed as third place.
  Fact 1.163: In 2015, Hawaii and Pfeiffer are listed as fifth place.
  Fact 1.164: In 2016, the championship was held in University Park, Pennsylvania (host Penn State) at Rec Hall.
  Fact 1.165: In 2016, Ohio State (31-2) won the final 3-0 over runner-up BYU (27-4).
  Fact 1.166: In 2016, UCLA and Long Beach State are listed as third place.
  Fact 1.167: In 2016, Erskine and George Mason are listed as fifth place.
  Fact 1.168: In 2017, the championship was held in Columbus, Ohio (host Ohio State) at St. John Arena.
  Fact 1.169: In 2017, Ohio State (32-2) won the final 3-0 over runner-up BYU (26-5).
  Fact 1.170: In 2017, Hawaii and Long Beach State are listed as third place.
  Fact 1.171: In 2017, Barton and Penn State are listed as fifth place.
  Fact 1.172: In 2018, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.
  Fact 1.173: In 2019, the championship was held in Long Beach, California (host Long Beach State) at The Pyramid.
  Fact 1.174: In 2020, the championship was held in Fairfax, Virginia (host George Mason) at EagleBank Arena.
  Fact 1.175: In 2021, the championship was held in Columbus, Ohio (host Ohio State) at Covelli Arena.
  Fact 1.176: In 2022, the championship was held in Los Angeles (host UCLA) at Pauley Pavilion.

Retrieved items:
<item 1>
NCAA Men's National Collegiate Volleyball Championship
The six teams were seeded according to the same methods used to seed the four teams in previous tournaments ; the top two seeds received byes into the Final
Four , while the third seed faced the sixth seed and the fourth seed faced the fifth seed in the quarterfinals . May 5 : Quarterfinals ( # 3 vs. # 6 seed ; # 4
vs. # 5 seed ) May 7 : Semifinals ( # 1 vs. # 4 - # 5 winner ; # 2 vs. # 3 - # 6 winner ) May 9 : NCAA Championship Quarterfinals Tuesday , May 5 , 2015 Maples
Pavilion , Stanford , California Semifinals Thursday , May 7 , 2015 Maples Pavilion , Stanford , California Championship Saturday , May 9 , 2015 Maples Pavilion
, Stanford ,
</item 1>

<item 2>
NCAA Men's National Collegiate Volleyball Championship
California - based universities have dominated the men 's volleyball national championship ; Loyola Chicago , Penn State , Ohio State and BYU are the only
non-California universities to have won the National Collegiate championship ; Hawaii and Lewis also won the championship tournament , but had their victories
vacated due to NCAA rules violations . Only seven non-California universities have participated in the National Collegiate championship match ( Loyola , BYU ,
Penn State , Ohio State , IPFW , Hawaii , and Lewis ) , although other universities such as Princeton and Ball State have participated in the final four . Only
three finals have involved two non-California schools : the 2003 final , when Lewis defeated BYU but had its win vacated ; the 2015 final , in which Loyola
defeated Lewis ; and the 2016 final , when Ohio State defeated BYU . UCLA , Southern California , Penn State , Stanford , and Long Beach State are the only
schools in Division I to have won an NCAA national championship in both men and women 's volleyball .
</item 2>

<item 3>
NCAA Men's National Collegiate Volleyball Championship
Gender Equity -- Boys ' and Mens ' Volleyball Participation Continues to Lag , April 2009 Jump up ^
http://fs.ncaa.org/Docs/stats/m_volleyball_champs_records/2012/d1/2012d1mvbchamp.pdf Jump up ^ UC Irvine defeats BYU in three sets to claim consecutive national
title , NCAA.com , May 5 , 2013 Jump up ^ Loyola repeats as men 's NCAA volleyball national champions , Chicago Tribune , May 10 , 2015 Jump up ^ NCAA Tourney
to have 2 play - in matches starting in 2014 , Off the Block , October 23 , 2013 NCAA Men 's National Collegiate Volleyball Championship Tournaments 1970 1971
1972 1973 1976 1977 1978 1979 1980 1981 1982 1984 1985 1986 1988 1989 1990 1991 1992 1993 1994 1995 1997 1998 1999 2000 2001 2002 2003 2005 2006
</item 3>

<item 4>
NCAA Men's National Collegiate Volleyball Championship
Champions ( edit ) NCAA Men 's National Collegiate Volleyball Championship Year Site ( Host ) Host Arena Final Third Place Final / Other participants Winner
Score Runner - up Third Place Score Fourth Place 1970 Details Los Angeles ( UCLA ) Pauley Pavilion UCLA ( 24 -- 1 ) 3 -- 0 Long Beach State UC Santa Barbara 2
-- 0 Ball State 1971 Details UCLA ( 29 -- 1 ) 3 -- 2 UC Santa Barbara Ball State 2 -- 0 Springfield 1972 Details Muncie , Indiana ( Ball State ) Irving
Gymnasium UCLA ( 27 -- 7 ) 3 -- 2 San Diego State Ball State 2 -- 0 UC Santa Barbara 1973 Details San Diego ( San Diego State ) Peterson Gym San Diego State (
21 -- 5 ) 3 -- 1 Long Beach State Ball State 2 -- 0 Army Details Santa Barbara ,
</item 4>

<item 5>
NCAA Men's National Collegiate Volleyball Championship
Pennsylvania ( Penn State ) Rec Hall Ohio State ( 26 -- 6 ) 3 -- 2 UC Santa Barbara ( 18 -- 15 ) Penn State and USC 2012 Details Los Angeles ( USC ) Galen
Center UC Irvine ( 26 -- 5 ) 3 -- 0 USC ( 24 -- 6 ) Lewis and Penn State 2013 Details Los Angeles ( UCLA ) Pauley Pavilion UC Irvine ( 25 -- 7 ) 3 -- 0 BYU ( 26
-- 5 ) Loyola Chicago and Penn State 2014 Details Chicago ( Loyola Chicago ) Gentile Arena Loyola Chicago ( 29 - 1 ) 3 - 1 Stanford ( 24 - 9 ) 3rd -- BYU and
Penn State 5th -- Lewis and Erskine 2015 Details Stanford , California ( Stanford ) Maples Pavilion Loyola Chicago ( 28 - 2 ) 3 - 2 Lewis ( 27 - 4 ) 3rd -- UC
Irvine and Penn State 5th -- Hawaii and Pfeiffer 2016 Details University Park ,
</item 5>

<item 6>
NCAA Men's National Collegiate Volleyball Championship
Pennsylvania ( Penn State ) Rec Hall Ohio State ( 31 - 2 ) 3 - 0 BYU ( 27 - 4 ) 3rd -- UCLA and Long Beach State 5th -- Erskine and George Mason 2017 Details
Columbus , Ohio ( Ohio State ) St. John Arena Ohio State ( 32 -- 2 ) 3 - 0 BYU ( 26 - 5 ) 3rd -- Hawaii and Long Beach State 5th -- Barton and Penn State 2018
Details Los Angeles ( UCLA ) Pauley Pavilion 2019 Details Long Beach , California ( Long Beach State ) The Pyramid 2020 Details Fairfax , Virginia ( George
Mason ) EagleBank Arena 2021 Details Columbus , Ohio ( Ohio State ) Covelli Arena 2022 Details Los Angeles ( UCLA ) Pauley Pavilion † Vacated due to NCAA
violations Team titles ( edit ) UCLA Pepperdine USC UC Irvine BYU Penn State Stanford Loyola Ohio State ⇙ Hawaii Lewis Long Beach State San Diego State Schools
with national championships -- 19 championships , -- 5 championships , -- 4 championships , -- 3 championships , -- 2 championships ,
</item 6>

<item 7>
NCAA Men's National Collegiate Volleyball Championship
-- 5 championships , -- 4 championships , -- 3 championships , -- 2 championships , -- 1 championship School # Year won UCLA 19 1970 , 1971 , 1972 , 1974 , 1975
, 1976 , 1979 , 1981 , 1982 , 1983 , 1984 , 1987 , 1989 , 1993 , 1995 , 1996 , 1998 , 2000 , 2006 Pepperdine 5 1978 , 1985 , 1986 , 1992 , 2005 Southern
California 1977 , 1980 , 1988 , 1990 UC Irvine 2007 , 2009 , 2012 , 2013 Ohio State 2011 , 2016 , 2017 BYU 1999 , 2001 , 2004 Penn State 1994 , 2008 Stanford
1997 , 2010 Loyola Chicago 2014 , 2015 Vacated 2002 , 2003 Long Beach State 1991 San Diego State 1973 Past tournaments ( edit ) Historically ,
</item 7>

<item 8>
NCAA Men's National Collegiate Volleyball Championship
Stanford , California Championship Saturday , May 9 , 2015 Maples Pavilion , Stanford , California Lewis ( 3 ) 25 22 25 25 Hawaii ( 1 ) 22 20 25 25 5 Penn State
( 1 ) 20 25 16 20 5 Penn State ( 3 ) 25 25 17 27 Lewis ( 2 ) 25 23 15 27 21 Loyola ( Chicago ) ( 3 ) 21 25 25 25 23 UC Irvine ( 0 ) 22 19 17 Loyola ( Chicago )
( 3 ) 25 33 25 Loyola ( Chicago ) ( 3 ) 25 25 25 6 Pfeiffer ( 0 ) 20 31 15 2016 ( edit ) Main article : 2016 NCAA Men 's National Collegiate Volleyball
Tournament See also ( edit ) NCAA Division III Men 's Volleyball Championship NCAA Women 's Volleyball Championship List of NCAA men 's volleyball schools
Pre-NCAA Intercollegiate Volleyball Champions Notes ( edit ) Jump up ^ Athletic Business ,
</item 8>

<item 9>
NCAA Men's National Collegiate Volleyball Championship
2000 2001 2002 2003 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2017 National Collegiate Athletic Association NCAA Awards Hall of Champions
Conferences Division I Institutions Athletic directors Baseball Championship CWS Basketball Men Women Cross Country Men Women Field hockey Championship Football
FBS FCS Golf Men Women Ice hockey Men Women Lacrosse Men Women Rowing Women Soccer Men Women Softball Championship WCWS Swimming & Diving Men Women Tennis Men
Women Track and Field Men 's indoor and outdoor Women 's indoor and outdoor Volleyball Men Women Wrestling Championship Division II Institutions Baseball
Championship Basketball Men Women Cross Country Men Women Field hockey Championship Football Championship Golf Men Women Gymnastics Men Women Ice hockey Men
Lacrosse Men Women Rowing Women Soccer Men Women Softball Championship / WCWS Swimming & Diving Men Women Tennis Men Women Track and Field Men 's indoor and
outdoor Women 's indoor and outdoor Volleyball Women Wrestling Championship Division III Institutions Baseball Championship Basketball Men Women Cross Country
Men Women Field hockey Championship Football Championship Golf Men Women Ice hockey Men Women Lacrosse Men Women Rowing Women Soccer Men Women Softball
</item 9>

<item 10>
NCAA Men's National Collegiate Volleyball Championship
( But for Hawaii 's vacated 2002 title , it would have matched this feat with 3 NCAA and 1 AIAW women 's national championships . ) In addition , Stanford (
1996 -- 97 ) and Penn State ( 2007 -- 08 ) are the only universities whose men and women 's volleyball programs won the national championship in the same
academic year . The 2016 championship tournament took place from May 3 -- 7 at Rec Hall on the campus of Penn State . 2011 ( edit ) May 5 , 2011 -- UC Santa
Barbara def . Southern California , 29 - 27 , 24 - 26 , 25 - 15 , 25 - 18 ; Ohio State def . Penn St. , 25 - 18 , 24 - 26 , 25 - 22 , 25 - 23 May 7 , 2011 --
Ohio State def .
</item 10>
=== END ITEM id=d63fa2b424080524 ===

=== ITEM id=04dc3e83cfddcd49 ===
Gold sentences and facts:
Sentence 0: People 's Postcode Lottery has a number of celebrity `` ambassadors '' . In March 2018 , these included David Attenborough , Clare Balding , Brian Blessed , Judi Dench , Shobna Gulati , Tim Healy , Stephen Jardine , Ellen MacArthur , Aggie MacKenzie , Carey Mulligan , John Stapleton and Emma Thompson . Fiona Phillips became an ambassador for People 's Postcode Lottery at the end of 2012 , leading the organisation 's DRTV campaigns .
  Fact 0.0: People's Postcode Lottery has a number of celebrity 'ambassadors'.
  Fact 0.1: In March 2018, the celebrity ambassadors of People's Postcode Lottery included David Attenborough.
  Fact 0.2: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Clare Balding.
  Fact 0.3: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Brian Blessed.
  Fact 0.4: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Judi Dench.
  Fact 0.5: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Shobna Gulati.
  Fact 0.6: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Tim Healy.
  Fact 0.7: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Stephen Jardine.
  Fact 0.8: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Ellen MacArthur.
  Fact 0.9: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Aggie MacKenzie.
  Fact 0.10: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Carey Mulligan.
  Fact 0.11: In March 2018, the celebrity ambassadors of People's Postcode Lottery included John Stapleton.
  Fact 0.12: In March 2018, the celebrity ambassadors of People's Postcode Lottery included Emma Thompson.
  Fact 0.13: Fiona Phillips became an ambassador for People's Postcode Lottery at the end of 2012.
  Fact 0.14: Fiona Phillips was leading People's Postcode Lottery's DRTV campaigns.

Retrieved items:
<item 1>
People's Postcode Lottery
People 's Postcode Lottery - wikipedia People 's Postcode Lottery Jump to : navigation , search This article does not cite any sources . Please help improve
this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( November 2017 ) ( Learn how and when to remove this
template message ) This article contains content that is written like an advertisement . Please help improve it by removing promotional content and
inappropriate external links , and by adding encyclopedic content written from a neutral point of view . ( January 2017 ) ( Learn how and when to remove this
template message ) People 's Postcode Lottery is a subscription lottery in Great Britain , run by Postcode Lottery Limited ( Gambling Commission licence numbers
000 - 000829 - N - 102511 and 000 - 000829 - R - 10251 ) .
</item 1>

<item 2>
Series 1 contestants and results: Tamara Beckwith (1st eliminated), Andi Peters (2nd), Andrea McLean (3rd), John Barrowman (4th), Sean Wilson (5th), Kelly
Holmes (6th), David Seaman (7th), Bonnie Langford (third), Stefan Booth (runner-up), and Gaynor Faye (winner). Series 2 began 20 January 2007 with 11
celebrities; its final was 17 March. Phillip
</item 2>

<item 3>
People's Postcode Lottery
Beneficiaries ( edit ) A minimum of 32 % goes to various trusts , which in turn provide funding to charities and community projects across Great Britain and
internationally . As an external lottery manager , People 's Postcode Lottery manages several society lotteries promoted by different charities . Over 4,000
causes have received support . Ambassadors ( edit ) People 's Postcode Lottery has a number of celebrity `` ambassadors '' . In March 2018 , these included
David Attenborough , Clare Balding , Brian Blessed , Judi Dench , Shobna Gulati , Tim Healy , Stephen Jardine , Ellen MacArthur , Aggie MacKenzie , Carey
Mulligan , John Stapleton and Emma Thompson . Fiona Phillips became an ambassador for People 's Postcode Lottery at the end of 2012 , leading the organisation
's DRTV campaigns . Presenters Judie McCourt , Danyl Johnson , Jeff Brazier and Matt Johnson visit People 's Postcode Lottery winners to present them with their
cheques , attend Postcode Million events and take part in People 's Postcode Lottery advertising campaigns .
</item 3>

<item 4>
People's Postcode Lottery
Postcode Challenge ( edit ) From 2008 to 2010 , People 's Postcode Lottery sponsored Postcode Challenge , a quiz show hosted by Angus Purden on STV .
Sponsorship ( edit ) Novamedia has worked with a number of leading TV production companies and public and commercial broadcasters to support the promotion of
its lotteries . Novamedia inspired and paid Endemol to produce TV game shows such as Deal or No Deal and 1 vs 100 .
</item 4>

<item 5>
People’s Postcode Lottery is a paid subscription lottery run by Postcode Lottery Limited in Great Britain, using players’ postcodes to determine winners.
Introduced by Novamedia, it launched in northern England in 2005, expanded to Scotland in 2007 and Wales in 2010. Players pay monthly and enter ten draws each
month; neighbours in a winning postcode may share prizes, subject to Gambling Act limits. At least 32% of proceeds goes to trusts supporting charities and
community projects,
</item 5>

<item 6>
People's Postcode Lottery
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 6>

<item 7>
The 2013 eighth series began 6 January. Beth Tweddle and Daniel Whiston won; Matt Lapinskas and Brianne Delcourt were runners-up, with Luke Campbell and Jenna
Smith third. Gareth Thomas and Robin Johnstone withdrew. The 2014 ninth, an All-Stars series, began 5 January. The preceding series’ top three were Matthew
Wolfenden (winner), Jorgie Porter (second) and Chico Slimani (third).
</item 7>

<item 8>
People's Postcode Lottery
In 2017 Novamedia BV threatened to sue the Free Postcode Lottery ( an alternative free service which raises money through advertising and offers daily prizes to
registered users ) on the basis that they were `` passing themselves off '' as the People 's Postcode Lottery . This claim was made in spite of the fact that
the free Postcode lottery was completely free while the Peoples Postcode lottery is a paid for subscription service . Process ( edit ) People 's Postcode
Lottery is a subscription lottery . Players sign up to pay monthly in advance by direct debit or debit card . There are ten draws each month with prizes every
day . A player 's tickets are based on their playing postcode , which is most commonly their home postcode . A given postcode in Britain will usually represent
a group of houses . If more than one resident has tickets in a winning postcode , all playing neighbours collect prizes . In accordance with restrictions under
the Gambling Act 2005 , the maximum amount which can be won by a single ticket is £ 400,000 or 10 % of the total draw proceeds .
</item 8>

<item 9>
People's Postcode Lottery
Contents ( hide ) 1 History 2 Process 3 Beneficiaries 4 Ambassadors 5 Postcode Challenge 6 Sponsorship 7 References 8 External links History ( edit ) The
Postcode Lottery format was introduced by the Dutch company Novamedia BV , which launched its first postcode lottery in the Netherlands in 1989 . Novamedia now
runs six lotteries using cause marketing : Nationale Postcode Loterij , Vrienden Loterij and BankGiro Loterij in the Netherlands , Svenska Postkod Lotteriet in
Sweden , People 's Postcode Lottery in Great Britain and , since July 2016 , Deutsche Postcode Lotterie in Germany . Together , they claim to have contributed
over € 7.9 billion to charitable organisations around the world . People 's Postcode Lottery was launched in the north of England in 2005 as a pilot scheme .
After a successful launch , it expanded into Scotland in 2007 and Wales in July 2010 . The lottery uses cause marketing , claiming to have been established to
raise money for charity and to increase awareness of the work of the various charities and causes supported by its players .
</item 9>

<item 10>
People's Postcode Lottery
References ( edit ) Jump up ^ Gambling Commission Account details Jump up ^ Novamedia Website , About Charity Lotteries section Jump up ^ People 's Postcode
Lottery - Charities webpage Jump up ^ People 's Postcode Lottery - Ambassadors webpage External links ( edit ) Official website Retrieved from ``
https://en.wikipedia.org/w/index.php?title=People%27s_Postcode_Lottery&oldid=836422769 '' Categories : Lotteries in the United Kingdom 2005 establishments in
the United Kingdom Hidden categories : All articles with unsourced statements Articles with unsourced statements from March 2018 Articles lacking sources from
November 2017 All articles lacking sources Articles with a promotional tone from January 2017 All articles with a promotional tone Use dmy dates from September
2014 Use British English from September 2014 Articles with unsourced statements from October 2012 Talk Contents About Wikipedia Add links This page was last
edited on 14 April 2018 , at 18 : 44 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using
this site , you agree to the Terms of Use and Privacy Policy .
</item 10>
=== END ITEM id=04dc3e83cfddcd49 ===

=== ITEM id=410f56872ee1b5b9 ===
Gold sentences and facts:
Sentence 0: The 2017 -- 18 NBA season is the 72nd season of the National Basketball Association ( NBA ) . The regular season began on October 17 , 2017 , earlier than previous seasons to reduce the number of `` back - to - back '' games teams are scheduled to play , with the 2017 Eastern Conference champion ( and Finals runner -- up ) Cleveland Cavaliers hosting a game against the Boston Celtics at Quicken Loans Arena in Cleveland , Ohio Christmas games were played on December 25 , 2017 . The 2018 NBA All - Star Game was played on February 18 , 2018 , at the Staples Center in Los Angeles , California . LeBron James of the Cleveland Cavaliers was named the All - Star Game Most Valuable Player . The regular season ended on April 11 , 2018 and the playoffs began on April 14 , 2018 .
  Fact 0.0: The 2017-18 NBA season is the 72nd season of the National Basketball Association (NBA).
  Fact 0.1: The 2017-18 NBA regular season began on October 17, 2017.
  Fact 0.2: The 2017-18 NBA regular season began earlier than previous seasons.
  Fact 0.3: The earlier start of the 2017-18 season was meant to reduce the number of 'back-to-back' games teams are scheduled to play.
  Fact 0.4: The 2017 Eastern Conference champion Cleveland Cavaliers, who were also Finals runner-up, hosted a game against the Boston Celtics at Quicken Loans Arena in Cleveland, Ohio.
  Fact 0.5: Christmas games were played on December 25, 2017.
  Fact 0.6: The 2018 NBA All-Star Game was played on February 18, 2018.
  Fact 0.7: The 2018 NBA All-Star Game was played at the Staples Center in Los Angeles, California.
  Fact 0.8: LeBron James of the Cleveland Cavaliers was named the All-Star Game Most Valuable Player.
  Fact 0.9: The 2017-18 NBA regular season ended on April 11, 2018.
  Fact 0.10: The 2017-18 NBA playoffs began on April 14, 2018.
Sentence 1: 2017 -- 18 NBA season League National Basketball Association Sport Basketball Duration October 17 , 2017 -- April 11 , 2018 April 14 -- May 28 , 2018 ( Playoffs ) May 31 -- June 17 , 2018 ( Finals ) Number of games 82 Number of teams 30 TV partner ( s ) ABC , TNT , ESPN , NBA TV Draft Top draft pick Markelle Fultz Picked by Philadelphia 76ers Regular season Top seed Houston Rockets Top scorer James Harden ( Houston ) Playoffs Finals NBA seasons ← 2016 -- 17 2018 -- 19 →
  Fact 1.0: The league of the 2017-18 NBA season is the National Basketball Association.
  Fact 1.1: The sport of the 2017-18 NBA season is basketball.
  Fact 1.2: The 2017-18 NBA regular season ran from October 17, 2017 to April 11, 2018.
  Fact 1.3: The 2017-18 NBA playoffs ran from April 14 to May 28, 2018.
  Fact 1.4: The 2017-18 NBA Finals ran from May 31 to June 17, 2018.
  Fact 1.5: The 2017-18 NBA season had 82 games.
  Fact 1.6: The 2017-18 NBA season had 30 teams.
  Fact 1.7: ABC, TNT, ESPN and NBA TV were the TV partners of the 2017-18 NBA season.
  Fact 1.8: Markelle Fultz was the top draft pick, picked by the Philadelphia 76ers.
  Fact 1.9: The Houston Rockets were the top seed in the 2017-18 NBA regular season.
  Fact 1.10: James Harden of the Houston Rockets was the top scorer of the 2017-18 NBA season.
  Fact 1.11: The 2016-17 season is the season before the 2017-18 NBA season.
  Fact 1.12: The 2018-19 season is the season after the 2017-18 NBA season.

Retrieved items:
<item 1>
2017–18 NBA season
Denver Minnesota Oklahoma City Portland Utah Pacific Golden State L.A. Clippers L.A. Lakers Phoenix Sacramento Southwest Dallas Houston Memphis New Orleans San
Antonio National Basketball Association Eastern Conference Atlantic Boston Celtics Brooklyn Nets New York Knicks Philadelphia 76ers Toronto Raptors Central
Chicago Bulls Cleveland Cavaliers Detroit Pistons Indiana Pacers Milwaukee Bucks Southeast Atlanta Hawks Charlotte Hornets Miami Heat Orlando Magic Washington
Wizards Western Conference Northwest Denver Nuggets Minnesota Timberwolves Oklahoma City Thunder Portland Trail Blazers Utah Jazz Pacific Golden State Warriors
Los Angeles Clippers Los Angeles Lakers Phoenix Suns Sacramento Kings Southwest Dallas Mavericks Houston Rockets Memphis Grizzlies New Orleans Pelicans San
Antonio Spurs Annual events Draft Eligibility Summer League Christmas Day All - Star Weekend Game Global Games Africa 2015 ,
</item 1>

<item 2>
2017–18 NBA season
The regular season started eight days earlier than the previous season . This move by the NBA aims for fewer `` back - to - back '' games , even fewer than the
previous season which had already reached an all - time low to help address teams that are `` resting '' their superstars . The NBA decreased the maximum number
of timeouts in a regulation game from 18 to 14 . The three team timeouts during the last two minutes of the game were also decreased to two per team in the last
three minutes to improve the flow and pace of the game . The trade deadline was moved from four days after the NBA All - Star Game to ten days before it , so
teams can settle their rosters before the All - Star break . Free - throw shooters will be assessed a delay - of - game violation if they venture outside of the
three - point line in between attempts . The NBA also added two - way contracted players , signifying an increasing partnership between the NBA and the newly
branded NBA G League . Referees can now assess dangerous closeouts of defenders to jump shooters with little space to land for a technical or a flagrant foul .
</item 2>

<item 3>
2017–18 NBA season
Africa 2017 Playoffs List Finals Champions History Predecessors BAA NBL ABA Merger Criticisms and controversies 2007 Tim Donaghy betting scandal Lockouts
Seasons Records regular season post-season All - Star Game Win - loss records Personalities Players Current rosters Foreign players Race and ethnicity First
overall draft picks Highest paid Retired numbers Banned or suspended NBPA Head coaches Current Player - coaches Champions Foreign coaches NBCA Owners Referees
Awards and honors Larry O'Brien Trophy NBA Awards NBA MVP Finals MVP All - Star Game MVP Hall of Fame Members NBA Silver Anniversary Team NBA 35th Anniversary
Team 50 Greatest Players Others Arenas Business Collective bargaining agreement Salary cap NBA Store Culture Cheerleading Mascots Dress code G League Midwest
Division Media TV NBA TV Rivalries Teams Defunct Expansion Relocated Timeline WNBA Basketball in the United States Category Portal 2017 -- 18 season Retrieved
from `` https://en.wikipedia.org/w/index.
</item 3>

<item 4>
Game seven
1984 Boston Celtics Los Angeles Lakers 111 -- 102 Boston Garden , Boston 1988 Los Angeles Lakers Detroit Pistons 108 -- 105 The Forum , Inglewood , California
1994 Houston Rockets New York Knicks 90 -- 84 The Summit , Houston 2005 San Antonio Spurs Detroit Pistons 81 -- 74 SBC Center , San Antonio Los Angeles Lakers
Boston Celtics 83 -- 79 Staples Center , Los Angeles Fifth time that Boston and L.A. met in a game seven . Kobe Bryant won his second consecutive Finals MVP
award , scoring 10 of his game high 23 points in the fourth quarter , as well as grabbing 15 rebounds to clinch the victory for L.A. 2013 Miami Heat San Antonio
Spurs 95 -- 88 American Airlines Arena , Miami Last time the NBA finals was a 2 -- 3 -- 2 format . 2016 Cleveland Cavaliers Golden State Warriors 93 -- 89
Oracle Arena , Oakland , California First time road team to win game seven of the NBA Finals since 1978 . Cavaliers became the first team to win the NBA Finals
after trailing 3 -- 1 in the series .
</item 4>

<item 5>
2017–18 NBA season
2017 -- 18 NBA season - wikipedia 2017 -- 18 NBA season Jump to : navigation , search 2017 -- 18 NBA season League National Basketball Association Sport
Basketball Duration October 17 , 2017 -- April 11 , 2018 April 14 -- May 28 , 2018 ( Playoffs ) May 31 -- June 17 , 2018 ( Finals ) Number of games 82 Number of
teams 30 TV partner ( s ) ABC , TNT , ESPN , NBA TV Draft Top draft pick Markelle Fultz Picked by Philadelphia 76ers Regular season Top seed Houston Rockets Top
scorer James Harden ( Houston ) Playoffs Finals NBA seasons ← 2016 -- 17 2018 -- 19 → The 2017 -- 18 NBA season is the 72nd season of the National Basketball
Association ( NBA ) .
</item 5>

<item 6>
Game seven
NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Minneapolis Makes One Last Title Run '' . NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Nats Win First
Title of Shot - Clock Era '' . NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Celtics Officially Launch Their Dynasty '' . NBA.com . Retrieved 2012 - 05 - 25
. Jump up ^ `` Hawks Force Game 7 , But Celtics Repeat '' . NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Celtics , Lakers Work OT to Start Rivalry '' .
NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` Red Lights Cigar to Close Coaching Career '' . NBA.com . Retrieved 2012 - 05 - 25 . Jump up ^ `` West 's
Heroics Ca n't Deprive Russell of 11th Title '' . NBA.com .
</item 6>

<item 7>
2017–18 NBA season
573 18.0 82 9 Denver Nuggets 46 36 . 561 19.0 82 10 Los Angeles Clippers 42 40 . 512 23.0 82 11 Los Angeles Lakers 35 47 . 427 30.0 82 12 Sacramento Kings 27 55
. 329 38.0 82 13 Dallas Mavericks 24 58 . 293 41.0 82 14 Memphis Grizzlies 22 60 . 268 43.0 82 15 Phoenix Suns 21 61 . 256 44.0 82 Notes z -- Clinched home
court advantage for the entire playoffs c -- Clinched home court advantage for the conference playoffs y -- Clinched division title x -- Clinched playoff spot *
-- Division leader Tiebreakers ( edit ) Eastern Conference Miami clinched # 6 seed over Milwaukee based on head - to - head sweep ( 3 -- 0 ) .
</item 7>

<item 8>
2017–18 NBA season
The regular season began on October 17 , 2017 , earlier than previous seasons to reduce the number of `` back - to - back '' games teams are scheduled to play ,
with the 2017 Eastern Conference champion ( and Finals runner -- up ) Cleveland Cavaliers hosting a game against the Boston Celtics at Quicken Loans Arena in
Cleveland , Ohio Christmas games were played on December 25 , 2017 . The 2018 NBA All - Star Game was played on February 18 , 2018 , at the Staples Center in
Los Angeles , California . LeBron James of the Cleveland Cavaliers was named the All - Star Game Most Valuable Player . The regular season ended on April 11 ,
2018 and the playoffs began on April 14 , 2018 . Contents ( hide ) 1 Transactions 1.1 Retirement 1.2 In - season trades 1.3 Free agency 1.4 Coaching changes
1.4. 1 Off - season 1.4. 2 In - season 2 Preseason 2.1 International games 3 Regular season 3.1 By conference 3.1.
</item 8>

<item 9>
Game seven
The NBA Finals has been consistently played in a best - of - seven series format since its inception . The games seven where the champions were awarded are :
Year Winning team Losing team Score Site Notes 1951 Rochester Royals New York Knicks 79 -- 75 Edgerton Park Arena , Rochester , New York New York forced a game
seven after trailing 0 -- 3 in the series . 1952 Minneapolis Lakers New York Knicks 82 -- 65 Minneapolis Auditorium , Minneapolis 1954 Minneapolis Lakers
Syracuse Nationals 87 -- 70 Minneapolis Auditorium , Minneapolis 1955 Syracuse Nationals Fort Wayne Pistons 92 -- 91 Onondaga County War Memorial , Syracuse ,
New York The home team won every game of this series . 1957 Boston Celtics St. Louis Hawks 125 -- 123 ( 2OT ) Boston Garden , Boston Only game seven in NBA
history to go into double overtime . 1960 Boston Celtics St.
</item 9>

<item 10>
2017–18 NBA season
92 1992 -- 93 1993 -- 94 1994 -- 95 1995 -- 96 1996 -- 97 1997 -- 98 1998 -- 99 2000s 1999 -- 00 2000 -- 01 2001 -- 02 2002 -- 03 2003 -- 04 2004 -- 05 2005 --
06 2006 -- 07 2007 -- 08 2008 -- 09 2010s 2009 -- 10 2010 -- 11 2011 -- 12 2012 -- 13 2013 -- 14 2014 -- 15 2015 -- 16 2016 -- 17 2017 -- 18 2018 -- 19 2017 --
18 NBA season by team Draft Summer League All - Star Game Playoffs Finals Transactions G League Eastern Atlantic Boston Brooklyn New York Philadelphia Toronto
Central Chicago Cleveland Detroit Indiana Milwaukee Southeast Atlanta Charlotte Miami Orlando Washington Western Northwest Denver Minnesota Oklahoma City
Portland Utah Pacific Golden State L.A. Clippers L.A.
</item 10>
=== END ITEM id=410f56872ee1b5b9 ===
