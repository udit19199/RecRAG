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

=== ITEM id=a37e13a230fd44d8 ===
Gold sentences and facts:
Sentence 0: In graph theory , a loop ( also called a self - loop or a `` buckle '' ) is an edge that connects a vertex to itself . A simple graph contains no loops .
  Fact 0.0: In graph theory, a loop is an edge that connects a vertex to itself
  Fact 0.1: A loop is also called a self-loop
  Fact 0.2: A loop is also called a buckle
  Fact 0.3: A simple graph contains no loops

Retrieved items:
<item 1>
Loop (graph theory)
ISBN 1 - 58488 - 291 - 3 . External links ( edit ) This article incorporates public domain material from the NIST document : Black , Paul E. `` Self loop '' .
Dictionary of Algorithms and Data Structures . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Loop_(graph_theory)&oldid=815655329 '' Categories :
Graph theory Talk Contents About Wikipedia Català Deutsch Español Esperanto Français Polski Português Русский Simple English Українська 3 more Edit links This
page was last edited on 16 December 2017 , at 05 : 43 . About Wikipedia
</item 1>

<item 2>
Loop (graph theory)
ISBN 0 - 07 - 005489 - 4 . Bollobás , Béla ; Modern Graph Theory , Springer ; 1st edition ( August 12 , 2002 ) . ISBN 0 - 387 - 98488 - 7 . Diestel , Reinhard ;
Graph Theory , Springer ; 2nd edition ( February 18 , 2000 ) . ISBN 0 - 387 - 98976 - 5 . Gross , Jonathon L , and Yellen , Jay ; Graph Theory and Its
Applications , CRC Press ( December 30 , 1998 ) . ISBN 0 - 8493 - 3982 - 0 . Gross , Jonathon L , and Yellen , Jay ; ( eds ) ; Handbook of Graph Theory . CRC (
December 29 , 2003 ) . ISBN 1 - 58488 - 090 - 2 . Zwillinger , Daniel ; CRC Standard Mathematical Tables and Formulae , Chapman & Hall / CRC ; 31st edition (
November 27 , 2002 ) .
</item 2>

<item 3>
Loop (graph theory)
Such a graph is called a bouquet . Contents ( hide ) 1 Degree 2 See also 3 References 4 External links Degree ( edit ) For an undirected graph , the degree of a
vertex is equal to the number of adjacent vertices . A special case is a loop , which adds two to the degree . This can be understood by letting each connection
of the loop edge count as its own adjacent vertex . In other words , a vertex with a loop `` sees '' itself as an adjacent vertex from both ends of the edge
thus adding two , not one , to the degree . For a directed graph , a loop adds one to the in degree and one to the out degree . See also ( edit ) Loops in Graph
Theory Cycle ( graph theory ) Graph theory Glossary of graph theory Loops in Topology Möbius ladder Möbius strip Strange loop Klein bottle References ( edit )
Balakrishnan , V.K. ; Graph Theory , McGraw - Hill ; 1 edition ( February 1 , 1997 ) . ISBN 0 - 07 - 005489 - 4 .
</item 3>

<item 4>
Loop (graph theory)
Loop ( graph Theory ) - wikipedia Loop ( graph Theory ) Jump to : navigation , search A graph with a loop on vertex 1 In graph theory , a loop ( also called a
self - loop or a `` buckle '' ) is an edge that connects a vertex to itself . A simple graph contains no loops . Depending on the context , a graph or a
multigraph may be defined so as to either allow or disallow the presence of loops ( often in concert with allowing or disallowing multiple edges between the
same vertices ) : Where graphs are defined so as to allow loops and multiple edges , a graph without loops or multiple edges is often distinguished from other
graphs by calling it a `` simple graph '' . Where graphs are defined so as to disallow loops and multiple edges , a graph that does have loops or multiple edges
is often distinguished from the graphs that satisfy these constraints by calling it a `` multigraph '' or `` pseudograph '' . In a graph with one vertex , all
edges must be loops . Such a graph is called a bouquet .
</item 4>

<item 5>
DNA
Here , the single - stranded DNA curls around in a long circle stabilized by telomere - binding proteins . At the very end of the T - loop , the single -
stranded telomere DNA is held onto a region of double - stranded DNA by the telomere strand disrupting the double - helical DNA and base pairing to one of the
two strands . This triple - stranded structure is called a displacement loop or D - loop . Single branch Multiple branches Branched DNA can form networks
containing multiple branches . Branched DNA Further information : Branched DNA and DNA nanotechnology In DNA , fraying occurs when non-complementary regions
exist at the end of an otherwise complementary double - strand of DNA . However , branched DNA can occur if a third strand of DNA is introduced and contains
adjoining regions able to hybridize with the frayed regions of the pre-existing double - strand . Although the simplest example of branched DNA involves only
three strands of DNA , complexes involving additional strands and multiple branches are also possible . Branched DNA can be used in nanotechnology to construct
geometric shapes , see the section on uses in technology below .
</item 5>

<item 6>
Vanishing point
She notes , in terms of projective geometry , the vanishing point is the image of the point at infinity associated with L , as the sightline from O through the
vanishing point is parallel to L . Vanishing line ( edit ) As a vanishing point originates in a line , so a vanishing line originates in a plane α that is not
parallel to the picture π . Given the eye point O , and β the plane parallel to α and lying on O , then the vanishing line of α is β ∩ π . For example , when α
is the ground plane and β is the horizon plane , then the vanishing line of α is the horizon line β ∩ π . Anderson notes , `` Only one particular vanishing line
occurs , often referred to as the `` horizon '' . To put it simply , the vanishing line of some plane , say α , is obtained by the intersection of the image
plane with another plane , say β , parallel to the plane of interest ( α ) , passing through the camera center . For different sets of lines parallel to this
plane α , their respective vanishing points will lie on this vanishing line .
</item 6>

<item 7>
Independence (probability theory)
Using this definition , it is easy to show that if X and Y are random variables and Y is constant , then X and Y are independent , since the σ - algebra
generated by a constant random variable is the trivial σ - algebra ( ∅ , Ω ) . Probability zero events can not affect independence so independence also holds if
Y is only Pr - almost surely constant . Properties ( edit ) Self - independence ( edit ) Note that an event is independent of itself if and only if P ( A ) = P
( A ∩ A ) = P ( A ) ⋅ P ( A ) ⇔ P ( A ) = 0 or 1 ( \ displaystyle \ mathrm ( P ) ( A ) = \ mathrm ( P ) ( A \ cap A ) = \ mathrm ( P ) ( A ) \ cdot \ mathrm ( P
) ( A ) \ Leftrightarrow \ mathrm ( P ) ( A ) = 0 ( \ text ( or ) ) 1 ) . Thus an event is independent of itself if and only if it almost surely occurs or its
complement almost surely occurs .
</item 7>

<item 8>
Vanishing point
5 D Bird 's - eye view Cross section Cutaway drawing Exploded view drawing Fisheye lens Panorama Worm 's - eye view Zoom lens Topics ( show ) 3D projection
Anamorphosis Axonometry Computer graphics Computer - aided design Descriptive geometry Engineering drawing Map projection Picture plane Plans ( drawings )
Projection ( linear algebra ) Projection plane Projective geometry Stereoscopy Technical drawing Vanishing point Video game graphics Viewing frustum Contents (
hide ) 1 Vector notation 2 Theorem 3 Vanishing line 4 Properties of vanishing points 5 Curvilinear and reverse perspective 6 Detection of vanishing points 7
Applications of vanishing points 8 See also 9 References 10 External links Vector notation ( edit ) A 2D construction of perspective viewing , showing the
formation of a vanishing point The vanishing point may also be referred to as the `` direction point '' , as lines having the same directional vector , say D ,
will have the same vanishing point or converge at the same vanishing points .
</item 8>

<item 9>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 9>

<item 10>
Vanishing point
Vanishing point - wikipedia Vanishing point Jump to : navigation , search For other uses , see Vanishing point ( disambiguation ) . A vanishing point can be
seen at the far end of this railroad . A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective
projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a
picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the
image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three
vanishing points . Part of a series on Graphical projection Planar ( show ) Parallel projection Orthographic projection Multiview projection Axonometric
projection Isometric projection Oblique projection Perspective projection Curvilinear perspective Reverse perspective Views ( show ) 2.
</item 10>
=== END ITEM id=a37e13a230fd44d8 ===

=== ITEM id=80388ca1f0e0e17c ===
Gold sentences and facts:
Sentence 0: `` The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit . In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
  Fact 0.0: The Glory of Love is a song.
  Fact 0.1: The Glory of Love was written by Billy Hill.
  Fact 0.2: The Glory of Love was recorded by Benny Goodman in 1936.
  Fact 0.3: Benny Goodman's version of The Glory of Love was a number one pop hit.
  Fact 0.4: In 1951, the R&B vocal group The Five Keys had their biggest R&B hit with their version of The Glory of Love.
  Fact 0.5: The Five Keys' version hit number one on the R&B chart for four non-consecutive weeks.
  Fact 0.6: The Five Keys' recording reportedly sold a million copies.
  Fact 0.7: Pressed recordings of The Five Keys' version are very rare.
Sentence 1: `` The Glory of Love '' Single by Benny Goodman Released 1936 Format 78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952
  Fact 1.0: The Glory of Love is a single by Benny Goodman.
  Fact 1.1: The Glory of Love was released in 1936.
  Fact 1.2: The Glory of Love has the format 78 rpm vinyl.
  Fact 1.3: The Glory of Love was recorded in 1936.
  Fact 1.4: The Glory of Love has the genre pop.
  Fact 1.5: The songwriter of The Glory of Love is Billy Hill.
  Fact 1.6: The Benny Goodman singles chronology lists The Glory of Love (1936).
  Fact 1.7: The Five Keys singles chronology lists The Glory of Love (1951).
  Fact 1.8: The Five Keys singles chronology lists Yes Sir, That's My Baby (1952).

Retrieved items:
<item 1>
Joy to the World (Three Dog Night song)
Sugar '' -- The Archies 1970 : `` Bridge over Troubled Water '' -- Simon & Garfunkel 1971 : `` Joy to the World '' -- Three Dog Night 1972 : `` The First Time
Ever I Saw Your Face '' -- Roberta Flack 1973 : `` Tie a Yellow Ribbon Round the Ole Oak Tree '' -- Tony Orlando and Dawn 1974 : `` The Way We Were '' -- Barbra
Streisand 1975 : `` Love Will Keep Us Together '' -- Captain & Tennille 1976 : `` Silly Love Songs '' -- Wings 1977 : `` Tonight 's the Night ( Gonna Be Alright
) '' -- Rod Stewart 1978 : `` Shadow Dancing '' -- Andy Gibb 1979 : `` My Sharona '' -- The Knack Complete list ( 1946 -- 1959 ) ( 1960 -- 1979 ) ( 1980 -- 1999
) ( 2000 -- 2019 ) Retrieved from `` https://en.wikipedia.org/w/index.
</item 1>

<item 2>
The Glory of Love (song)
Although The Five Keys recording sold a reported million copies pressed recordings are very rare . Contents ( hide ) 1 Charts 1.1 Weekly charts 2 Otis Redding
version 3 Charts 3.1 Weekly charts 4 References Charts ( edit ) Weekly charts ( edit ) Chart ( 1951 ) Peak position US Billboard Hot Rhythm & Blues Songs Otis
Redding version ( edit ) `` The Glory of Love '' Single by Otis Redding from the album The Dock of the Bay B - side `` I 'm Coming Home '' Released 1967 Format
7 '' vinyl Recorded December , 1967 Genre Soul , Pop Length 2 : 38 Label Volt S419 Songwriter ( s ) Billy Hill Producer ( s ) Steve Cropper Otis Redding singles
chronology `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 ) `` Shake '' ( 1967 ) `` The Glory of Love '' ( 1967 ) `` Tramp '' ( 1967 )
In 1967 , Otis Redding recorded a cover version for his 1968 album , The Dock of the Bay .
</item 2>

<item 3>
The Glory of Love (song)
In 1951 , R&B vocal group , The Five Keys , had their biggest R&B hit with their version of the song , hitting number one on the R&B chart for four
non-consecutive weeks . Although The Five Keys recording sold a reported million copies pressed recordings are very rare .
</item 3>

<item 4>
Let My Love Open the Door
Billboard Hot 100 59 U.S. Cashbox 92 Personnel ( edit ) Pete Townshend : vocals , guitars , synths John `` Rabbit '' Bundrick : `` straight '' keyboards Simon
Phillips : drums Tony Butler : bass guitar Ted Jensen - mastering engineer Covers ( edit ) A version of this song by Luminate appears on The Covers EP . It was
also covered by M. Ward ( for the 2005 Starbucks compilation album , Sweetheart 2005 : Love Songs ) Christian rock group Audio Adrenaline covered this song on
their 1999 album Underdog . Natalie Imbruglia covered the song on her album Male released in 2015 . Great Big Sea covered the song on their 2012 album XX .
Roger Daltrey covered the song in 2015 for a campaign involving Teen Cancer America and First Citizens Bank . A full version was released in February 2016 ,
with all proceeds going to Teen Cancer America . The group Rogue Wave also covered it in `` Cover me '' released in 2017 .
</item 4>

<item 5>
Let My Love Open the Door
Initially , Townshend 's manager despised the track due to it `` not sounding like Townshend , '' and wanted it to be removed from Empty Glass . However , upon
the song 's chart success , his manager called to apologize . Despite the song 's critical and commercial success , Pete Townshend did not consider it one of
his best songs . He told Rolling Stone in an interview that `` Let My Love Open the Door '' was `` just a ditty , '' also claiming that he preferred his minor
US hit `` A Little Is Enough '' from the same album . In 1996 , Townshend released a new version of the song , called `` the E. Cola mix '' , turning the song
into a ballad . This version appeared in different television shows and film soundtracks . Chart performance ( edit ) Weekly charts ( edit ) Chart ( 1980 ) Peak
position Canada RPM 100 5 UK Singles Chart 46 U.S. Billboard Hot 100 9 U.S. Cash Box Top 100 11 Year - end charts ( edit ) Chart ( 1980 ) Rank Canada 35 U.S.
</item 5>

<item 6>
The Glory of Love (song)
Billboard . Retrieved 2012 - 11 - 22 . Jump up ^ `` Billboard R&B Singles Chart '' . Billboard . Retrieved 2016 - 01 - 27 . Preceded by `` Sixty Minute Man ''
by The Dominoes Billboard Best Selling Retail Rhythm & Blues Records number - one single ( The Five Keys version ) September 22 , 1951 Succeeded by `` Sixty
Minute Man '' by The Dominoes Retrieved from `` https://en.wikipedia.org/w/index.php?title=The_Glory_of_Love_(song)&oldid=814898703 '' Categories : 1936 songs
1951 singles Benny Goodman songs Otis Redding songs Songs written by Billy Hill ( songwriter ) Billboard Hot R&B / Hip - Hop Songs number - one singles Paul
McCartney songs Hidden categories : Articles needing additional references from December 2015 All articles needing additional references Music infoboxes with
deprecated parameters Articles with hAudio microformats Talk Contents About Wikipedia Norsk Edit links This page was last edited on 11 December 2017 , at 15 :
51 . About Wikipedia
</item 6>

<item 7>
The Glory of Love (song)
The Glory of Love ( song ) - wikipedia The Glory of Love ( song ) Jump to : navigation , search This article needs additional citations for verification .
Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2015 ) ( Learn how and
when to remove this template message ) For the song by Peter Cetera , see Glory of Love . `` The Glory of Love '' Single by Benny Goodman Released 1936 Format
78 rpm vinyl Recorded 1936 Genre Pop Songwriter ( s ) Billy Hill Benny Goodman singles chronology `` The Glory of Love ' '' ( 1936 ) The Glory of Love'1936 The
Five Keys singles chronology `` The Glory of Love ' '' ( 1951 ) The Glory of Love'1951 `` Yes Sir , That 's My Baby '' ( 1952 ) Yes Sir , That 's My Baby1952 ``
The Glory of Love '' is a song written by Billy Hill , recorded by Benny Goodman in 1936 , whose version was a number one pop hit .
</item 7>

<item 8>
The Glory of Love (song)
Redding 's cover became a top 20 hit and reached number nineteen on the Billboard R&B Songs chart and number sixty on the Billboard Hot 100 . Charts ( edit )
Weekly charts ( edit ) Chart ( 1967 ) Peak position US Billboard Hot 100 60 US Billboard Hot Rhythm & Blues Songs 19 References ( edit ) Jump up ^ Whitburn ,
Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record Research . p. 205 . Jump up ^ Propes , Steve ( 1973 ) . Those Oldies But Goodies : A Guide to
50 's Record Collecting . The Macmillan Company , New York . p. 42 . Jump up ^ Whitburn , Joel ( 2004 ) . Top R&B / Hip - Hop Singles : 1942 - 2004 . Record
Research . p. 486 . Jump up ^ `` allmusic ( ( ( Otis Redding > Awards ) ) ) '' . Billboard . Retrieved 2012 - 11 - 22 .
</item 8>

<item 9>
Graph query row:
work: The Glory of Love
songwriter: Billy Hill
relationship: CREATED
role: None
</item 9>

<item 10>
Let My Love Open the Door
Imdb.com. 2017 - 05 - 12 . Retrieved 2018 - 03 - 30 . External links ( edit ) Lyrics of this song at MetroLyrics ( hide ) Pete Townshend Discography Studio
albums Who Came First Rough Mix ( w / Ronnie Lane ) Empty Glass All the Best Cowboys Have Chinese Eyes White City : A Novel The Iron Man : A Musical
Psychoderelict Classic Quadrophenia Meher Baba tribute albums ( w / Ronnie Lane et al . ) Happy Birthday I Am With Love O ' Parvardigar Live albums Deep End
Live !
</item 10>
=== END ITEM id=80388ca1f0e0e17c ===

=== ITEM id=5ad525be42f47f83 ===
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
( Karachi - XIII ) NA - 252 ( Karachi - XIV ) NA - 253 ( Karachi - XV ) NA - 254 ( Karachi - XVI ) NA - 255 ( Karachi - XVII ) NA - 256 ( Karachi - XVIII ) NA -
257 ( Karachi - XIX ) NA - 258 ( Karachi - XX ) National Assembly constituencies from Balochistan ( NA - 259 to NA - 272 ) NA - 259 ( Quetta ) NA - 260 ( Quetta
- cum - Chagai - cum - Mastung ) NA - 261 ( Pishin - cum - Ziarat ) NA - 262 ( Killa Abdullah ) NA - 263 ( Loralai ) NA - 264 ( Zhob - cum - Killa Saifullah )
NA - 265 ( Sibi - cum - Kohlu - cum - Dera Bugti ) NA - 266 ( Nasirabad ) NA - 267 ( Kachhi ) NA - 268 ( Kalat - cum - Mastung ) NA - 269 ( Khuzdar ) NA - 270 (
Awaran - cum - Lasbela ) NA - 271 ( Kharan -
</item 1>

<item 2>
National Assembly of Pakistan
631 0.47 % 0 9 National Peoples Party 197,829 0.44 % 0 10 Pakistan Muslim League ( Q ) 1,409,905 3.11 % 0 0 11 Awami National Party 453,057 1.00 % 0 0 12 Qaumi
Watan Party ( Sherpao ) 46,574 0.10 % 0 0 13 Pakistan Muslim League ( Z ) 128,510 0.28 % 0 0 14 National Party 61,148 0.13 % 0 0 15 Awami Muslim League Pakistan
93,046 0.20 % 0 0 16 Awami Jamhuri Ittehad Pakistan 71,175 0.16 % 0 0 17 Balochistan National Party 63,979 0.14 % 0 0 18 All Pakistan Muslim League 54,231 0.12
% 0 0 19 Independents 5,880,658 12.96 % 27 8 0 0 8 Total 45,388,
</item 2>

<item 3>
National Assembly of Pakistan
The committees have also been empowered to invite or summon before it any member or any other person having a special interest in relation to any matter under
its consideration and may hear expert evidence and hold public hearing . Composition and elections ( edit ) The composition of the National Assembly is
specified in Article 51 of the Constitution of Pakistan . There are a total of 332 seats in the National Assembly . Of these , 272 are filled by direct
elections . In addition , the Pakistani Constitution reserves 10 seats for religious minorities and 60 seats for women , to be filled by proportional
representation among parties with more than 5 % of the vote . As of 2006 , there are 72 women members in the Assembly . Members of the National Assembly are
elected by the people in competitive multi-party elections , to be held at most five years apart on universal adult franchise . To be a member of electoral
college , according to Article 62 of the Constitution , candidates must be citizens of Pakistan and not less than 18 years of age . 2013 election ( edit ) Main
article : Pakistani general election , 2013 Results of the Pakistani general election ,
</item 3>

<item 4>
National Assembly of Pakistan
( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP ( 3 ) ANP ( 2 ) PML -- Q ( 2 ) AJIP ( 1 ) AML ( 1 ) APML ( 1 ) BNP ( 1 ) PML -- Z ( 1 ) QWP ( 1 ) Ind ( 10 ) Elections Voting
system Mixed member majoritarian ( First past the post for most seats , 60 seats reserved for women and 10 seats reserved for religious minorities by
proportional representation ) Last election 11 May 2013 Meeting place Parliament House , Islamabad Website www.na.gov.pk The Pakistani National Assembly ( Urdu
: قومی اسمبلئ پاکستان ‬ ‎ -- Qaumī Asimbli'e Pākistān ) ; is the lower house of the bicameral Majlis - e-Shura , which also comprises the President of Pakistan
and Senate ( upper house ) . The National Assembly and the Senate both convene at Parliament House in Islamabad .
</item 4>

<item 5>
National Assembly of Pakistan
96 % 27 8 0 0 8 Total 45,388,404 100 % 266 266 60 10 336 Withheld : NA - 103 , NA - 237 ; Postponed : NA - 38 ; Re-poll ordered : NA - 46 ; Terminated : NA - 83
, NA - 254 Source : Detailed Results General Elections - 2013 , Reserve Seats Women , Reserve Seats Minorities Dissolution ( edit ) National Assembly can be
dissolved at the initiative of the Prime Minister . If dissolved , new elections are conducted for the Assembly . Article 58 of the Constitution of Pakistan
deals with the dissolution of the Assembly : 58 . Dissolution of the National Assembly : ( 1 ) The President shall dissolve the National Assembly if so advised
by the Prime Minister ; and the National Assembly shall , unless sooner dissolved , stand dissolved at the expiration of forty - eight hours after the Prime
Minister has so advised .
</item 5>

<item 6>
National Assembly of Pakistan
National Assembly of Pakistan - wikipedia National Assembly of Pakistan Jump to : navigation , search For List of MNAs elected in Pakistani general election ,
2013 , see List of MNAs elected in Pakistani general election , 2013 . National Assembly of Pakistan قومی اسمبلئ پاکستان ‬ 14th National Assembly of Pakistan
Type Type Lower house of the Parliament of Pakistan Leadership Speaker Ayaz Sadiq ( PML - N ) Since 9 November 2015 Deputy Speaker Javed Abbasi ( PML - N )
Since 3 June 2013 Leader of House Shahid Khaqan Abbasi ( PML - N ) Since 1 August 2017 Leader of Opposition Khurshid Shah ( PPP ) Since 7 June 2013 Structure
Seats 332 ( 3 vacant ) Political groups Government Coalition ( 209 ) PML -- N ( 188 ) JUI -- F ( 13 ) PML -- F ( 5 ) NPP ( 2 ) NP ( 1 ) Opposition ( 130 ) PPP (
47 ) PTI ( 32 ) MQM ( 24 ) JI ( 4 ) PkMAP (
</item 6>

<item 7>
National Assembly of Pakistan
- 26 ( Bannu ) NA - 27 ( Lakki Marwat ) NA - 28 ( Buner ) NA - 29 ( Swat - I ) NA - 30 ( Swat - II ) NA - 31 ( Shangla ) NA - 32 ( Chitral ) NA - 33 ( Upper Dir
- cum - Lower Dir ) NA - 34 ( Lower Dir ) NA - 35 ( Malakand ) National Assembly constituencies from Islamabad and FATA ( NA - 36 to NA - 49 ) NA - 36 ( Tribal
Area - I ) NA - 37 ( Tribal Area - II ) NA - 38 ( Tribal Area - III ) NA - 39 ( Tribal Area - IV ) NA - 40 ( Tribal Area - V ) NA - 41 ( Tribal Area - VI ) NA -
42 ( Tribal Area - VII ) NA - 43 ( Tribal Area - VIII ) NA - 44 ( Tribal Area - IX ) NA - 45 ( Tribal Area - X ) NA - 46 ( Tribal Area - XI ) NA - 47 ( Tribal
Area - XII ) NA - 48 ( Islamabad - I ) NA - 49 ( Islamabad -
</item 7>

<item 8>
National Assembly of Pakistan
The National Assembly and the Senate both convene at Parliament House in Islamabad . The National Assembly is a democratically elected body consisting of a
total of 332 members who are referred to as Members of the National Assembly ( MNAs ) , of which 272 are directly elected members and 70 reserved seats for
women and religious minorities . A political party must secure 172 seats to obtain and preserve a majority . Members are elected through the first - past - the
- post system under universal adult suffrage , representing electoral districts known as National Assembly constituencies . According to the constitution , the
70 seats reserved for women and religious minorities are allocated to the political parties according to their proportional representation . Each National
Assembly is formed for a five - year term , commencing from the date of the first sitting , after which it is automatically dissolved . Currently the National
Assembly can not be dissolved by the President of Pakistan , it is dissolved by the Prime Minister of Pakistan . Election for 13th National Assembly was held on
18 February 2008 . On March 17 , 2013 13th National Assembly was dissolved on completion of its five - year term under Article 52 of the Constitution .
</item 8>

<item 9>
National Assembly of Pakistan
Seat distribution ( edit ) In the 14th Assembly , the seats have been allocated among the four provinces , federally - administered area , and capital territory
of Pakistan as follows : Province General Women Total Punjab 148 35 183 Sindh 61 14 75 KPK 35 8 43 Balochistan 14 17 FATA 12 - 12 Federal Capital - See also (
edit ) Speaker of National Assembly Senate of Pakistan Politics of Pakistan Prime Minister of Pakistan Constitution of Pakistan President of Pakistan Supreme
Court of Pakistan Pakistan Bar Council Punjab Bar Council Supreme Court Bar Association of Pakistan List of Pakistan National Assembly Seats List of provincial
governments of Pakistan showing provincial assemblies Notes ( edit ) ^ Jump up to : `` PML - N 's Ayaz Sadiq elected Speaker , Murtaza Abbasi Deputy Speaker in
NA '' . Dawn News . 3 June 2013 . Retrieved 6 June 2013 . Jump up ^ `` PPP 's Khursheed Shah declared leader of opposition in NA '' . Dawn News . 7 June 2013 .
Retrieved 7 June 2013 .
</item 9>

<item 10>
National Assembly of Pakistan
The Public Accounts Committee has a special role to review the report of the Auditor General . Senate , the upper house of the Parliament , has equal
representation from the federating units balancing the provincial inequality in the National Assembly , where the number of members is based on population of
the provinces . The Senate 's role is to promote national cohesion and harmony , and work as a stabilizing factor of the federation . The Senate numbers a total
of 104 members who serve six - year terms which are alternated so that half the senators are up for re-election by the electoral college every three years . The
National Assembly consists of 332 members . The Constitution empowers the President to dissolve the National Assembly , but the Senate is not subject to
dissolution . Only the Parliament can amend the Constitution , by two - thirds majority vote separately in each House . Members of National Assembly ( edit )
See also : List of Pakistan National Assembly Seats Qualifications ( edit ) The Constitution of Pakistan lists a number of requirements for members of the
National Assembly in Article 62 .
</item 10>
=== END ITEM id=5ad525be42f47f83 ===

=== ITEM id=706ca1df308f9f92 ===
Gold sentences and facts:
Sentence 0: In probability theory , two events are independent , statistically independent , or stochastically independent if the occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does not affect the probability distribution of the other .
  Fact 0.0: In probability theory, two events can be called independent.
  Fact 0.1: In probability theory, two events can be called statistically independent.
  Fact 0.2: In probability theory, two events can be called stochastically independent.
  Fact 0.3: Two events are independent if the occurrence of one does not affect the probability of occurrence of the other.
  Fact 0.4: Two random variables are independent if the realization of one does not affect the probability distribution of the other.

Retrieved items:
<item 1>
Independence (probability theory)
A finite set of events is mutually independent if every event is independent of any intersection of the other events -- that is , if and only if for every k -
element subset of ( A i ) i = 1 n ( \ displaystyle \ ( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) , P ( ⋂ i = 1 k A i ) = ∏ i = 1 k P ( A i ) . ( \ displaystyle \
mathrm ( P ) \ left ( \ bigcap _ ( i = 1 ) ^ ( k ) A_ ( i ) \ right ) = \ prod _ ( i = 1 ) ^ ( k ) \ mathrm ( P ) ( A_ ( i ) ) . ) This is called the
multiplication rule for independent events . Note that it is not a single condition involving only the product of all the probabilities of all single events (
see below for a counterexample ) ; it must hold true for all subsets of events .
</item 1>

<item 2>
Independence (probability theory)
) and yet no two of the three events are pairwise independent ( and hence the set of events are not mutually independent ) . This example shows that mutual
independence involves requirements on the products of probabilities of all combinations of events , not just the single events as in this example . See also (
edit ) Copula ( statistics ) Independent and identically distributed random variables Mutually exclusive events Subindependence Linear dependence between random
variables Conditional independence Normally distributed and uncorrelated does not imply independent Mean dependence References ( edit ) Jump up ^ Russell ,
Stuart ; Norvig , Peter ( 2002 ) . Artificial Intelligence : A Modern Approach . Prentice Hall . p. 478 . ISBN 0 - 13 - 790395 - 2 . ^ Jump up to : Feller , W (
1971 ) . `` Stochastic Independence '' . An Introduction to Probability Theory and Its Applications . Wiley . Jump up ^ George , Glyn , `` Testing for the
independence of three events , '' Mathematical Gazette 88 , November 2004 , 568 .
</item 2>

<item 3>
Independence (probability theory)
Thus , the occurrence of B does not affect the probability of A , and vice versa . Although the derived expressions may seem more intuitive , they are not the
preferred definition , as the conditional probabilities may be undefined if P ( A ) or P ( B ) are 0 . Furthermore , the preferred definition makes clear by
symmetry that when A is independent of B , B is also independent of A . More than two events ( edit ) A finite set of events ( A i ) i = 1 n ( \ displaystyle \
( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) is pairwise independent if every pair of events is independent -- that is , if and only if for all distinct pairs of
indices m , k , P ( A m ∩ A k ) = P ( A m ) P ( A k ) . ( \ displaystyle \ mathrm ( P ) ( A_ ( m ) \ cap A_ ( k ) ) = \ mathrm ( P ) ( A_ ( m ) ) \ mathrm ( P )
( A_ ( k ) ) . )
</item 3>

<item 4>
Independence (probability theory)
More than two random variables ( edit ) A set of random variables is pairwise independent if and only if every pair of random variables is independent . Even if
the set of random variables is pairwise independent , it is not necessarily mutually independent as defined next . A set of random variables is mutually
independent if and only if for any finite subset X 1 , ... , X n ( \ displaystyle X_ ( 1 ) , \ ldots , X_ ( n ) ) and any finite sequence of numbers a 1 , ... ,
a n ( \ displaystyle a_ ( 1 ) , \ ldots , a_ ( n ) ) , the events ( X 1 ≤ a 1 ) , ... , ( X n ≤ a n ) ( \ displaystyle \ ( X_ ( 1 ) \ leq a_ ( 1 ) \ ) , \ ldots
, \ ( X_ ( n ) \ leq a_ ( n ) \ ) ) are mutually independent events ( as defined above ) . The measure - theoretically inclined may prefer to substitute events
( X ∈ A ) for events ( X ≤ a ) in the above definition , where A is any Borel set .
</item 4>

<item 5>
Independence means one event’s occurrence does not change another’s probability: \(P(A\cap B)=P(A)P(B)\). For random variables, their joint distribution factors
into their marginal distributions. With multiple events or variables, pairwise independence does not necessarily imply mutual independence; mutual independence
requires the product rule for every subset. Conditional independence means variables are independent once another variable is known. Independent variables have
zero covariance, but zero covariance alone does not imply independence. Examples: separate die rolls are independent
</item 5>

<item 6>
Independence (probability theory)
Independence ( probability theory ) - wikipedia Independence ( probability theory ) Jump to : navigation , search Part of a series on Statistics Probability
theory Probability axioms Probability space Sample space Elementary event Event Random variable Probability measure Complementary event Joint probability
Marginal probability Conditional probability Independence Conditional independence Law of total probability Law of large numbers Bayes ' theorem Boole 's
inequality Venn diagram Tree diagram In probability theory , two events are independent , statistically independent , or stochastically independent if the
occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does
not affect the probability distribution of the other . The concept of independence extends to dealing with collections of more than two events or random
variables , in which case the events are pairwise independent if each pair are independent of each other , and the events are mutually independent if each event
is independent of each other combination of events . Contents ( hide ) 1 Definition 1.1 For events 1.1. 1 Two events 1.1. 2 More than two events 1.2 For random
variables 1.2. 1 Two random variables 1.2.
</item 6>

<item 7>
Independence (probability theory)
Drawing cards ( edit ) If two cards are drawn with replacement from a deck of cards , the event of drawing a red card on the first trial and that of drawing a
red card on the second trial are independent . By contrast , if two cards are drawn without replacement from a deck of cards , the event of drawing a red card
on the first trial and that of drawing a red card on the second trial are not independent , because a deck that has had a red card removed has proportionately
fewer red cards . Pairwise and mutual independence ( edit ) Pairwise independent , but not mutually independent , events . Mutually independent events .
Consider the two probability spaces shown . In both cases , P ( A ) = P ( B ) = 1 / 2 and P ( C ) = 1 / 4 .
</item 7>

<item 8>
Independence (probability theory)
A ( \ displaystyle ( \ mathcal ( A ) ) ) and B ( \ displaystyle ( \ mathcal ( B ) ) ) are said to be independent if , whenever A ∈ A ( \ displaystyle A \ in ( \
mathcal ( A ) ) ) and B ∈ B ( \ displaystyle B \ in ( \ mathcal ( B ) ) ) , P ( A ∩ B ) = P ( A ) P ( B ) . ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \
mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) . )
</item 8>

<item 9>
Independence (probability theory)
1 Two random variables 1.2. 2 More than two random variables 1.2. 3 Conditional independence 1.3 Independent σ - algebras 2 Properties 2.1 Self - independence
2.2 Expectation and covariance 2.3 Characteristic function 3 Examples 3.1 Rolling dice 3.2 Drawing cards 3.3 Pairwise and mutual independence 3.4 Mutual
independence 4 See also 5 References Definition ( edit ) For events ( edit ) Two events ( edit ) Two events A and B are independent ( often written as A ⊥ B ( \
displaystyle A \ perp B ) or A ⊥ ⊥ B ( \ displaystyle A \ perp \ ! \ ! \ ! \ perp B ) ) if their joint probability equals the product of their probabilities : P
( A ∩ B ) = P ( A ) P ( B ) ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \ mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) ) .
</item 9>

<item 10>
Independence (probability theory)
For more than two events , a mutually independent set of events is ( by definition ) pairwise independent ; but the converse is not necessarily true ( see below
for a counterexample ) . For random variables ( edit ) Two random variables ( edit ) Two random variables X and Y are independent if and only if ( iff ) the
elements of the π - system generated by them are independent ; that is to say , for every a and b , the events ( X ≤ a ) and ( Y ≤ b ) are independent events (
as defined above ) .
</item 10>
=== END ITEM id=706ca1df308f9f92 ===

=== ITEM id=7270dd7d82dc079d ===
Gold sentences and facts:
Sentence 0: The book became an instant hit , and the online version received about 20 million views as of 2009 ; many online readers requested a printed version . Kinney had agreed , and in April 2007 , Diary of a Wimpy Kid was published . To date , twelve Wimpy Kid books have been released , plus one do - it - yourself book and two movie diaries . In April 2009 , TIME named Kinney one of The World 's Most Influential People .
  Fact 0.0: The book became an instant hit.
  Fact 0.1: The online version of the book received about 20 million views as of 2009.
  Fact 0.2: Many online readers requested a printed version of the book.
  Fact 0.3: Kinney agreed to a printed version of the book.
  Fact 0.4: Diary of a Wimpy Kid was published in April 2007.
  Fact 0.5: To date, twelve Wimpy Kid books have been released.
  Fact 0.6: To date, one do-it-yourself Wimpy Kid book has been released.
  Fact 0.7: To date, two Wimpy Kid movie diaries have been released.
  Fact 0.8: In April 2009, TIME named Kinney one of The World's Most Influential People.
Sentence 1: # Title Date ISBN Diary of a Wimpy Kid April 1 , 2007 ISBN 978 - 0 - 8109 - 9313 - 6 The introduction to the series introduces the protagonist , Greg Heffley , his best friend Rowley , his middle school problems , Halloween , and the Cheese Touch . Rodrick Rules February 1 , 2008 ISBN 978 - 0 - 8109 - 9473 - 7 Greg must stop Rodrick from revealing the secrets that had happened last summer . The Last Straw January 13 , 2009 ISBN 978 - 0 - 8109 - 7068 - 7 When Greg 's father threatens to send Greg to a military academy , Greg must find a way to prevent this from happening . Dog Days October 12 , 2009 ISBN 978 - 0 - 8109 - 8391 - 5 Summer is here , but when the family has not that much money to go to the beach ( money is tight at that time ) , it might not be the best summer ever . 5 The Ugly Truth November 9 , 2010 ISBN 978 - 0 - 8109 - 8491 - 2 Greg is growing up , and he will have to face the challenges of this period ( like changing dentists , and a school lock - in ) . 6 Cabin Fever November 15 , 2011 ISBN 978 - 1 - 4197 - 0223 - 5 The family is stranded in the house as snow begins to fall . 7 The Third Wheel November 13 , 2012 ISBN 978 - 1 - 4197 - 0584 - 7 The Valentine 's Day dance is coming up , and Greg has no one go with . But when Michael Sampson ( Abigail 's date ) will not be going , things turn around . 8 Hard Luck November 5 , 2013 ISBN 978 - 1 - 4197 - 1132 - 9 Greg 's friend count is dropping as Rowley now has a girlfriend , named Abigail ( continuing the last book ) . 9 The Long Haul November 4 , 2014 ISBN 978 - 1 - 4197 - 1189 - 3 This is the family 's second summer trip , but instead , it 's a road trip . But with the Heffleys , you know what to expect from this disaster . 10 Old School November 3 , 2015 ISBN 978 - 1 - 4197 - 1701 - 7 The neighborhood is going to cut electricity for the weekend , and as the family 's car has a fender bender , it forces Greg to go to the farm trip . Will Greg survive the electricity - free days ? 11 Double Down November 1 , 2016 ISBN 978 - 1 - 4197 - 2344 - 5 Greg 's mom wants him to do something else than video games , so he makes a movie in collaboration with Rowley and joins the music program . 12 The Getaway November 7 , 2017 ISBN 978 - 1 - 4197 - 2545 - 6 Instead of celebrating the holidays at home , Greg and his family head on a tropical vacation to a resort . But what first starts as an relaxing trip , later turns into a Christmas nightmare . Will Greg be able to turn this vacation around ?
  Fact 1.0: Diary of a Wimpy Kid was released on April 1, 2007, with ISBN 978-0-8109-9313-6.
  Fact 1.1: The introduction to the Diary of a Wimpy Kid series introduces the protagonist Greg Heffley, his best friend Rowley, his middle school problems, Halloween, and the Cheese Touch.
  Fact 1.2: Rodrick Rules was released on February 1, 2008, with ISBN 978-0-8109-9473-7.
  Fact 1.3: In Rodrick Rules, Greg must stop Rodrick from revealing the secrets that happened last summer.
  Fact 1.4: The Last Straw was released on January 13, 2009, with ISBN 978-0-8109-7068-7.
  Fact 1.5: In The Last Straw, when Greg's father threatens to send Greg to a military academy, Greg must find a way to prevent this.
  Fact 1.6: Dog Days was released on October 12, 2009, with ISBN 978-0-8109-8391-5.
  Fact 1.7: In Dog Days, summer is here, but the family does not have much money to go to the beach, so it might not be the best summer ever.
  Fact 1.8: The Ugly Truth was released on November 9, 2010, with ISBN 978-0-8109-8491-2.
  Fact 1.9: In The Ugly Truth, Greg is growing up and will have to face the challenges of this period, like changing dentists and a school lock-in.
  Fact 1.10: Cabin Fever was released on November 15, 2011, with ISBN 978-1-4197-0223-5.
  Fact 1.11: In Cabin Fever, the family is stranded in the house as snow begins to fall.
  Fact 1.12: The Third Wheel was released on November 13, 2012, with ISBN 978-1-4197-0584-7.
  Fact 1.13: In The Third Wheel, the Valentine's Day dance is coming up and Greg has no one to go with.
  Fact 1.14: In The Third Wheel, things turn around when Michael Sampson, Abigail's date, will not be going.
  Fact 1.15: Hard Luck was released on November 5, 2013, with ISBN 978-1-4197-1132-9.
  Fact 1.16: In Hard Luck, Greg's friend count is dropping as Rowley now has a girlfriend, named Abigail.
  Fact 1.17: Hard Luck continues the story from The Third Wheel.
  Fact 1.18: The Long Haul was released on November 4, 2014, with ISBN 978-1-4197-1189-3.
  Fact 1.19: In The Long Haul, this is the family's second summer trip, and it is a road trip.
  Fact 1.20: Old School was released on November 3, 2015, with ISBN 978-1-4197-1701-7.
  Fact 1.21: In Old School, the neighborhood is going to cut electricity for the weekend.
  Fact 1.22: In Old School, the family's car has a fender bender, which forces Greg to go on the farm trip.
  Fact 1.23: Old School asks whether Greg will survive the electricity-free days.
  Fact 1.24: Double Down was released on November 1, 2016, with ISBN 978-1-4197-2344-5.
  Fact 1.25: In Double Down, Greg's mom wants him to do something other than video games, so he makes a movie in collaboration with Rowley.
  Fact 1.26: In Double Down, Greg joins the music program.
  Fact 1.27: The Getaway was released on November 7, 2017, with ISBN 978-1-4197-2545-6.
  Fact 1.28: In The Getaway, instead of celebrating the holidays at home, Greg and his family head on a tropical vacation to a resort.
  Fact 1.29: In The Getaway, what first starts as a relaxing trip later turns into a Christmas nightmare.
  Fact 1.30: The Getaway asks whether Greg will be able to turn the vacation around.
Sentence 2: Diary of a Wimpy Kid The logo of the series , appearing on the book covers , blogs , web comic , and the films . Diary of a Wimpy Kid Diary of a Wimpy Kid : Rodrick Rules Diary of a Wimpy Kid : The Last Straw Diary of a Wimpy Kid : Dog Days Diary of a Wimpy Kid : The Ugly Truth Diary of a Wimpy Kid : Cabin Fever Diary of a Wimpy Kid : The Third Wheel Diary of a Wimpy Kid : Hard Luck Diary of a Wimpy Kid : The Long Haul Diary of a Wimpy Kid : Old School Diary of a Wimpy Kid : Double Down Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Genre Realistic fiction Publisher Scholastic Publishing / Amulet Books / Puffin Books Published April 1 , 2007 -- present Media type Print ( hardback & paperback ) Audiobook E-book
  Fact 2.0: The logo of the Diary of a Wimpy Kid series appears on the book covers, blogs, web comic, and films.
  Fact 2.1: The Diary of a Wimpy Kid series includes the books Rodrick Rules, The Last Straw, Dog Days, The Ugly Truth, Cabin Fever, The Third Wheel, Hard Luck, The Long Haul, Old School, Double Down, and The Getaway.
  Fact 2.2: The Diary of a Wimpy Kid series was written by Jeff Kinney.
  Fact 2.3: The Diary of a Wimpy Kid series is from the United States.
  Fact 2.4: The Diary of a Wimpy Kid series is in English.
  Fact 2.5: The Diary of a Wimpy Kid series is in the realistic fiction genre.
  Fact 2.6: The Diary of a Wimpy Kid series is published by Scholastic Publishing, Amulet Books, and Puffin Books.
  Fact 2.7: The Diary of a Wimpy Kid series was published from April 1, 2007 to the present.
  Fact 2.8: The Diary of a Wimpy Kid series is available in print (hardback and paperback), as an audiobook, and as an e-book.

Retrieved items:
<item 1>
Diary of a Wimpy Kid (book series)
Diary of a Wimpy Kid ( book series ) - Wikipedia Diary of a Wimpy Kid ( book series ) This is the latest accepted revision , reviewed on 22 April 2018 . Jump to
: navigation , search `` Diary of a Wimpy Kid ( series ) '' redirects here . For the film series , see Diary of a Wimpy Kid ( film series ) . Diary of a Wimpy
Kid The logo of the series , appearing on the book covers , blogs , web comic , and the films .
</item 1>

<item 2>
Diary of a Wimpy Kid (book series)
ISBN 0 - 8109 - 9313 - 9 . ^ Jump up to : `` '' Diary of a Wimpy Kid ( 2010 ) '' on `` . Comingsoon.net . March 19 , 2010 . Retrieved May 7 , 2011 . ^ Jump up
to : `` Diary of a Wimpy Kid '' . FunBrain . Family Education Network . Retrieved March 5 , 2011 . Jump up ^ `` WIMPY 2 LAST '' . 3 November 2015 . Retrieved 6
March 2016 . Jump up ^ The 2009 TIME 100 : Jeff Kinney Time magazine 's 2009 list of The World 's Most Influential People Jump up ^ Kinney , Jeff . `` The Wimpy
Kid Movie Diary : The Next Chapter '' . Retrieved March 28 , 2017 . Jump up ^ Nickelodeon Kids ' Choice Awards , Nick.com Jump up ^ Dixler , Elsa . `` The New
York Times Best Sellers : Children 's Books '' . The New York Times .
</item 2>

<item 3>
Diary of a Wimpy Kid (book series)
Contents ( hide ) 1 Background 2 Chronology 2.1 Supplementary books 2.2 Toys and games 2.3 Audiobooks 3 Characters 4 Achievements 4.1 Awards and honors 4.2
Commercial success 5 Films 6 Musical 7 References 8 External links Background ( edit ) In January 1998 , Diary of a Wimpy Kid officially began , when Kinney
brought up an idea about a middle - school weakling named Greg Heffley , who wrote about his personal life . In May 2004 , FunBrain and Kinney released an
online version of Diary of a Wimpy Kid . The website made daily entries until June 2005 . He worked on his book for almost eight years , before showing it to a
publisher in New York . In February 2006 , during the New York Comic Con , Jeff signed a multi book deal with publisher Harry N. Abrams , Inc. , to turn Diary
of a Wimpy Kid into a print series .
</item 3>

<item 4>
Diary of a Wimpy Kid (book series)
All the main books are the journals of the main character , Greg Heffley . Befitting a teen 's diary , the books are filled with hand - written notes and simple
drawings of Greg 's daily adventures . Since the release of the online version in May 2004 , most of the books have garnered positive reviews and commercial
success . As of February 2008 , it has been purchased one million times . The first , second , fourth , and ninth installments have been adapted into films by
20th Century Fox . The website Poptropica released islands in the theme of Diary of a Wimpy Kid called Wimpy Wonderland and Wimpy Boardwalk , both of which
credit Kinney in an ambiguous contributor 's role .
</item 4>

<item 5>
Diary of a Wimpy Kid (book series)
2007 ) Diary of a Wimpy Kid : Rodrick Rules ( February 1 , 2008 ) Diary of a Wimpy Kid : The Last Straw ( January 13 , 2009 ) Diary of a Wimpy Kid : Dog Days (
October 12 , 2009 ) Diary of a Wimpy Kid : The Ugly Truth ( November 9 , 2010 ) Diary of a Wimpy Kid : Cabin Fever ( November 15 , 2011 ) Diary of a Wimpy Kid :
The Third Wheel ( November 13 , 2012 ) Diary of a Wimpy Kid : Hard Luck ( November 5 , 2013 ) Diary of a Wimpy Kid : The Long Haul ( November 4 , 2014 ) Diary
of a Wimpy Kid : Old School ( November 3 , 2015 ) Diary of a Wimpy Kid : Double Down ( November 1 , 2016 ) Diary of a Wimpy Kid : The Getaway ( November 7 ,
</item 5>

<item 6>
Diary of a Wimpy Kid (book series)
Diary of a Wimpy Kid Diary of a Wimpy Kid : Rodrick Rules Diary of a Wimpy Kid : The Last Straw Diary of a Wimpy Kid : Dog Days Diary of a Wimpy Kid : The Ugly
Truth Diary of a Wimpy Kid : Cabin Fever Diary of a Wimpy Kid : The Third Wheel Diary of a Wimpy Kid : Hard Luck Diary of a Wimpy Kid : The Long Haul Diary of a
Wimpy Kid : Old School Diary of a Wimpy Kid : Double Down Diary of a Wimpy Kid : The Getaway Author Jeff Kinney Country United States Language English Genre
Realistic fiction Publisher Scholastic Publishing / Amulet Books / Puffin Books Published April 1 , 2007 -- present Media type Print ( hardback & paperback )
Audiobook E-book Diary of a Wimpy Kid is a series of fiction books written by the American author and cartoonist Jeff Kinney . All the main books are the
journals of the main character , Greg Heffley .
</item 6>

<item 7>
Diary of a Wimpy Kid (book series)
USA Today , which ranks best sellers based on sales alone regardless of genre or intended audience , has listed Diary of a Wimpy Kid in its top 150 , peaking at
number 8 , for 141 weeks as of April 4 , 2010 . As of the same date , Rodrick Rules has been on the list for 117 weeks ( peaking at number 4 ) , The Last Straw
has been on the list for 65 weeks ( peaking at number 1 ) , Dog Days has been on the list for all 25 weeks of its publication ( peaking at number 1 ) , and The
Wimpy Kid Movie Diary has been listed for all three weeks of its publication , peaking at number 2 . The series has sold over 194 million books in more than 30
countries . The series has also generated over $500 million in revenue . Films ( edit ) Main article : Diary of a Wimpy Kid ( film series ) This section needs
to be updated . Please update this article to reflect recent events or newly available information .
</item 7>

<item 8>
Diary of a Wimpy Kid (book series)
The Wimpy Kid Movie Diary 3 : Dog Days is the second updated version of the first book , now adding stills from the third film Dog Days , and information on how
this film was made . The Wimpy Kid Movie Diary : The Next Chapter covers how the film The Long Haul was made . It is dedicated to Jason Drucker , the actor who
played Greg in that movie . Toys and games ( edit ) Action figures included a minifigure pack ( with Greg , Rowley , and Fregley ) , a 6 - inch Greg Heffley
figure , a 6 - inch holiday figure based on Cabin Fever , and a Greg and Rodrick plush doll . Audiobooks ( edit ) Twelve audiobooks have been released for the
Diary of a Wimpy Kid series . The first six are read by Ramon de Ocampo and Dan Russell from number seven . They are presented by Recorded Books and distributed
by Audible Inc . Diary of a Wimpy Kid ( April 1 , 2007 ) Diary of a Wimpy Kid : Rodrick Rules ( February 1 ,
</item 8>

<item 9>
Diary of a Wimpy Kid (book series)
Will Greg be able to turn this vacation around ? Supplementary books ( edit ) Diary of a Wimpy Kid Do - It - Yourself Book ( published between Rodrick Rules and
The Last Straw ) is an activity book that features a blank section encouraging readers to keep their own journal in the style of the books . The Wimpy Kid Do -
It - Yourself Book : Extended Edition ( published after Wimpy Kid Movie Diary 2 and before Cabin Fever ) is the same book as the first Do - It - Yourself Book ,
but with extra 60 pages and 16 more full - color comics . The Wimpy Kid Movie Diary is a book about the making of the first film , which features stills and
brand - new illustrations . Jeff Kinney dedicated the book to the film 's main stars , Zachary Gordon and Robert Capron . The Wimpy Kid Movie Diary 2 : Rodrick
Rules is an updated version of the first book , now featuring stills from the second film Rodrick Rules , more new illustrations , and information on how the
second film was made .
</item 9>

<item 10>
Diary of a Wimpy Kid (book series)
, to turn Diary of a Wimpy Kid into a print series . The book became an instant hit , and the online version received about 20 million views as of 2009 ; many
online readers requested a printed version . Kinney had agreed , and in April 2007 , Diary of a Wimpy Kid was published . To date , twelve Wimpy Kid books have
been released , plus one do - it - yourself book and two movie diaries . In April 2009 , TIME named Kinney one of The World 's Most Influential People .
Chronology ( edit ) # Title Date ISBN Diary of a Wimpy Kid April 1 , 2007 ISBN 978 - 0 - 8109 - 9313 - 6 The introduction to the series introduces the
protagonist , Greg Heffley , his best friend Rowley , his middle school problems , Halloween , and the Cheese Touch . Rodrick Rules February 1 , 2008 ISBN 978 -
0 - 8109 - 9473 - 7 Greg must stop Rodrick from revealing the secrets that had happened last summer .
</item 10>
=== END ITEM id=7270dd7d82dc079d ===

=== ITEM id=18e3e4005cc7fe39 ===
Gold sentences and facts:
Sentence 0: Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi. tɕíŋ ) ( listen ) ) , formerly romanized as Peking , is the capital of the People 's Republic of China , the world 's second most populous city proper , and most populous capital city . The city , located in northern China , is governed as a direct - controlled municipality under the national government with 16 urban , suburban , and rural districts . Beijing Municipality is surrounded by Hebei Province with the exception of neighboring Tianjin Municipality to the southeast ; together the three divisions form the Jingjinji metropolitan region and the national capital region of China .
  Fact 0.0: Beijing is the capital of the People's Republic of China.
  Fact 0.1: Beijing was formerly romanized as Peking.
  Fact 0.2: Beijing is the world's second most populous city proper.
  Fact 0.3: Beijing is the most populous capital city.
  Fact 0.4: Beijing is located in northern China.
  Fact 0.5: Beijing is governed as a direct-controlled municipality under the national government.
  Fact 0.6: Beijing has 16 urban, suburban, and rural districts.
  Fact 0.7: Beijing Municipality is surrounded by Hebei Province, except for the neighboring Tianjin Municipality to the southeast.
  Fact 0.8: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the Jingjinji metropolitan region.
  Fact 0.9: Beijing Municipality, Hebei Province, and Tianjin Municipality together form the national capital region of China.
Sentence 1: Over the past 3,000 years , the city of Beijing has had numerous other names . The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University , still use the former romanization .
  Fact 1.0: Over the past 3,000 years, the city of Beijing has had numerous other names.
  Fact 1.1: The name Beijing means 'Northern Capital'.
  Fact 1.2: The name Beijing is made of the Chinese characters 北 (north) and 京 (capital).
  Fact 1.3: The name Beijing was applied to the city in 1403 during the Ming dynasty.
  Fact 1.4: The name Beijing was applied to distinguish the city from Nanjing, the 'Southern Capital'.
  Fact 1.5: The English spelling Beijing is based on the pinyin romanization of the two characters as pronounced in Standard Mandarin.
  Fact 1.6: The older English spelling Peking is the postal romanization of the same two characters as pronounced in Chinese dialects spoken in the southern port towns.
  Fact 1.7: The southern port towns were the first visited by European traders and missionaries.
  Fact 1.8: The Chinese dialects spoken in the southern port towns preserve the Middle Chinese pronunciation of 京 as kjaeng.
  Fact 1.9: That pronunciation of 京 as kjaeng comes before a phonetic shift in the northern dialects to the modern pronunciation.
  Fact 1.10: Peking is no longer the common name for the city.
  Fact 1.11: Some of the city's older locations and facilities still use the former romanization Peking.
  Fact 1.12: Beijing Capital International Airport, with IATA code PEK, still uses the former romanization.
  Fact 1.13: Peking University still uses the former romanization.

Retrieved items:
<item 1>
Beijing
1928 -- 1937 ; 1945 -- 1949 ) Literal meaning `` Northern Capital '' ( show ) Transcriptions Standard Mandarin Hanyu Pinyin Běijīng Bopomofo ㄅ ㄟ ˇ ㄐ ㄧ ㄥ Gwoyeu
Romatzyh Beeijing Wade -- Giles Pei - ching IPA ( pèi. tɕíŋ ) Wu Suzhounese Poh - cin Hakka Romanization Bet - gin Yue : Cantonese Yale Romanization Bākgìng or
Bākgīng IPA ( pɐ́k̚. kêŋ ) or ( pɐ́k̚. kéŋ ) Jyutping Bak1ging1 Southern Min Hokkien POJ Pak - kian Tâi - lô Pak - kiann Eastern Min Fuzhou BUC Báe̤k - gĭng
Beijing ( / beɪˈdʒɪŋ / ; Mandarin : ( pèi.
</item 1>

<item 2>
Beijing
Retrieved 22 July 2009 . External links Find more aboutBeijingat Wikipedia 's sister projects Definitions from Wiktionary Media from Wikimedia Commons News from
Wikinews Quotations from Wikiquote Texts from Wikisource Textbooks from Wikibooks Travel guide from Wikivoyage Learning resources from Wikiversity Economic
profile for Beijing at HKTDC Visit Beijing Facebook Page Photograph of The approach to Peking -- outside the walls taken in 1890 by Sir Henry Norman Preceded by
Lin'an ( Song dynasty ) Capital of China ( as Dadu of Yuan ) 1264 -- 1368 Succeeded by Nanjing ( Ming dynasty ) Preceded by Nanjing ( Ming dynasty ) Capital of
China 1420 -- 1928 Succeeded by Nanjing ( ROC ) Preceded by Nanjing ( ROC ) Capital of the People 's Republic of China 1949 -- present Succeeded by present
capital Places adjacent to Beijing Zhangjiakou , Hebei Chengde , Hebei Chengde , Hebei Zhangjiakou , Hebei Beijing Langfang , Hebei Baoding , Hebei Langfang ,
</item 2>

<item 3>
Beijing
And under the Hongwu Emperor 's feudal policies it was given to Zhu Di , one of his sons , who was created `` Prince of Yan '' . The early death of Zhu
Yuanzhang 's heir led to a succession struggle on his death , one that ended with the victory of Zhu Di and the declaration of the new Yongle era . Since his
harsh treatment of the Ming capital Yingtian ( modern Nanjing ) alienated many there , he established his fief as a new co-capital . The city of Beiping became
Beijing ( northern capital ) or Shuntian in 1403 . The construction of the new imperial residence , the Forbidden City , took from 1406 to 1420 ; this period
was also responsible for several other of the modern city 's major attractions , such as the Temple of Heaven and Tian'anmen . On 28 October 1420 , the city was
officially designated the capital of the Ming dynasty in the same year that the Forbidden City was completed . Beijing became the empire 's primary capital and
Yingtian , also called Nanjing ( southern capital ) , became the co-capital .
</item 3>

<item 4>
Beijing
The official Latin alphabet abbreviation for Beijing is `` BJ '' . History Main article : History of Beijing Early History The earliest traces of human
habitation in the Beijing municipality were found in the caves of Dragon Bone Hill near the village of Zhoukoudian in Fangshan District , where Peking Man lived
. Homo erectus fossils from the caves date to 230,000 to 250,000 years ago . Paleolithic Homo sapiens also lived there more recently , about 27,000 years ago .
Archaeologists have found neolithic settlements throughout the municipality , including in Wangfujing , located in downtown Beijing . The first walled city in
Beijing was Jicheng , the capital city of the state of Ji and was built in 1045 BC . Within modern Beijing , Jicheng was located around the present Guang'anmen
area in the south of Xicheng District . This settlement was later conquered by the state of Yan and made its capital . Early Imperial China The Tianning Pagoda
, built around 1120 during the Liao dynasty . After the First Emperor unified China , Jicheng became a prefectural capital for the region .
</item 4>

<item 5>
Beijing
Two generations later , Kublai Khan ordered the construction of Dadu ( or Daidu to the Mongols , commonly known as Khanbaliq ) , a new capital for his Yuan
dynasty to the northeast of the Zhongdu ruins . The construction took from 1264 to 1293 , but greatly enhanced the status of a city on the northern fringe of
China proper . The city was centered on the Drum Tower slightly to the north of modern Beijing and stretched from the present - day Chang'an Avenue to the
northern part of Line 10 subway . Remnants of the Yuan rammed earth wall still stand and are known as the Tucheng . Ming dynasty One of the corner towers of the
Forbidden City , which was built by Emperor Yongle in Ming dynasty . In 1368 , soon after declaring the new Hongwu era of the Ming dynasty , the rebel leader
Zhu Yuanzhang sent an army to Khanbaliq and conquered it . Since the Yuan continued to occupy Shangdu and Mongolia , however , Dadu was renamed to Beiping as
used to supply the military garrisons in the area .
</item 5>

<item 6>
Beijing
The name Beijing , which means `` Northern Capital '' ( from the Chinese characters 北 for north and 京 for capital ) , was applied to the city in 1403 during the
Ming dynasty to distinguish the city from Nanjing ( the `` Southern Capital '' ) . The English spelling is based on the pinyin romanization of the two
characters as they are pronounced in Standard Mandarin . An older English spelling , Peking , is the postal romanization of the same two characters as they are
pronounced in Chinese dialects spoken in the southern port towns first visited by European traders and missionaries . Those dialects preserve the Middle Chinese
pronunciation of 京 as kjaeng , prior to a phonetic shift in the northern dialects to the modern pronunciation . Although Peking is no longer the common name for
the city , some of the city 's older locations and facilities , such as Beijing Capital International Airport , with IATA Code PEK , and Peking University ,
still use the former romanization . The single Chinese character abbreviation for Beijing is 京 , which appears on automobile license plates in the city . The
official Latin alphabet abbreviation for Beijing is `` BJ '' .
</item 6>

<item 7>
Beijing
On 28 June the same year , Beijing 's name was returned to Beiping ( written at the time as `` Peiping '' ) . On July 7 , 1937 , the 29th Army and the Japanese
army in China exchanged fire at the Marco Polo Bridge near the Wanping Fortress southwest of the city . The Marco Polo Bridge Incident triggered the Second Sino
- Japanese War , World War II as it is known in China . During the war , Beiping fell to Japan on 29 July 1937 and was made the seat of the Provisional
Government of the Republic of China , a puppet state that ruled the ethnic - Chinese portions of Japanese - occupied northern China . This government was later
merged into the larger Wang Jingwei government based in Nanjing . People 's Republic of China Mao Zedong proclaiming the establishment of the People 's Republic
of China in 1949 In the final phases of the Chinese Civil War , the People 's Liberation Army seized control of the city peacefully on 31 January 1949 in the
course of the Pingjin Campaign .
</item 7>

<item 8>
Beijing
In 938 , after the fall of the Tang , the Later Jin ceded the entire northern frontier to the Khitan Liao dynasty , which treated the city as Nanjing , or the
`` Southern Capital '' , one of four secondary capitals to complement its `` Supreme Capital '' , Shangjing ( modern Baarin Left Banner in Inner Mongolia ) .
Some of the oldest surviving structures in Beijing date to the Liao period , including the Tianning Pagoda . Miaoying Temple and its White Stupa in Dadu which
was built by Kublai Khan The Liao fell to the Jurchen Jin dynasty in 1122 , which gave the city to the Song dynasty and then retook it in 1125 during its
conquest of northern China . In 1153 , the Jurchen Jin made Beijing their `` Central Capital '' , or Zhongdu . The city was besieged by Genghis Khan 's invading
Mongolian army in 1213 and razed to the ground two years later .
</item 8>

<item 9>
Beijing
Beijing - wikipedia Beijing Jump to : navigation , search `` Peking '' redirects here . For other uses , see Beijing ( disambiguation ) and Peking (
disambiguation ) . Beijing 北京 市 Municipality Beijing Municipality Clockwise from top : Beijing CBD skyline , Tiananmen , Temple of Heaven , National Center for
the Performing Arts , and Beijing National Stadium Location of Beijing Municipality within China Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383
° E ﻿ / 39.917 ; 116.383 Coordinates : 39 ° 55 ′ N 116 ° 23 ′ E ﻿ / ﻿ 39.917 ° N 116.383 ° E ﻿ / 39.917 ; 116.383 Country China Established date 1045 BCE ( Zhou
Dynasty ) Divisions - County - level - Township - level 16 districts 289 towns and villages Government Type Municipality Party Secretary Cai Qi Mayor Chen
Jining Congress Chairman Li Wei ( zh ) Conference Chairman Ji Lin Area Municipality 16,411 km ( 6,336 sq mi ) Land 16,
</item 9>

<item 10>
Beijing
After the First Emperor unified China , Jicheng became a prefectural capital for the region . During the Three Kingdoms period , it was held by Gongsun Zan and
Yuan Shao before falling to Cao Cao 's Wei Kingdom . The AD 3rd - century Western Jin demoted the town , placing the prefectural seat in neighboring Zhuozhou .
During the Sixteen Kingdoms period when northern China was conquered and divided by the Wu Hu , Jicheng was briefly the capital of the Xianbei Former Yan
Kingdom . After China was reunified during the Sui dynasty , Jicheng , also known as Zhuojun , became the northern terminus of the Grand Canal . Under the Tang
dynasty , Jicheng as Youzhou , served as a military frontier command center . During the An - Shi Rebellion and again amidst the turmoil of the late Tang ,
local military commanders founded their own short - lived Yan dynasties and called the city Yanjing , or the `` Yan Capital . '' Also in the Tang dynasty , the
city 's name Jicheng was replaced by Youzhou or Yanjing .
</item 10>
=== END ITEM id=18e3e4005cc7fe39 ===
