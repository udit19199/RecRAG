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

=== ITEM id=e23bb75b7bca1b45 ===
Gold sentences and facts:
Sentence 0: Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
  Fact 0.0: The glenoid cavity is relatively shallow compared to the acetabulum, which is at the hip joint.
  Fact 0.1: The shallow glenoid cavity makes the shoulder joint prone to dislocation (luxation).
  Fact 0.2: Strong glenohumeral ligaments and muscles prevent dislocation in most cases.

Retrieved items:
<item 1>
Glenoid cavity
Glenoid cavity - wikipedia Glenoid cavity Jump to : navigation , search Glenoid cavity Costal surface of left scapula . Glenoid cavity shown in red . Glenoid
cavity shown in red . This cavity articulates with the head of the humerus . Details Identifiers Latin cavitas glenoidalis , fossa glenoidalis Dorlands /
Elsevier c_16 / 12220465 TA A02. 4.01. 019 FMA 23275 Anatomical terms of bone ( edit on Wikidata ) The glenoid cavity or glenoid fossa of scapula is a part of
the shoulder . It is a shallow , pyriform articular surface , which is located on the lateral angle of the scapula . It is directed laterally and forward and
articulates with the head of the humerus ; it is broader below than above and its vertical diameter is the longest . This cavity forms the glenohumeral joint
along with the humerus . This type of joint is classified as a synovial , ball and socket joint .
</item 1>

<item 2>
Glenoid cavity
This type of joint is classified as a synovial , ball and socket joint . The humerus is held in place within the glenoid cavity by means of the long head of the
biceps tendon . This tendon originates on the superior margin of the glenoid cavity and loops over the shoulder , bracing humerus against the cavity . The
rotator cuff also reinforces this joint more specifically with the supraspinatus tendon to hold the head of the humerus in the glenoid cavity . The cavity
surface is covered with cartilage in the fresh state ; and its margins , slightly raised , give attachment to a fibrocartilaginous structure , the glenoid
labrum , which deepens the cavity . This cartilage is very susceptible to tearing . When torn , it is most commonly known as a SLAP lesion which is generally
caused by repetitive shoulder movements . Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder
joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
</item 2>

<item 3>
Glenoid cavity
at 16 : 19 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 3>

<item 4>
Glenoid cavity
edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online , SUNY Downstate Medical Center Mechanics of Glenohumeral Instability at University of Washington
Department of Orthopaedics ( hide ) Bones of the arm Shoulder girdle , clavicle conoid tubercle trapezoid line costal tuberosity subclavian groove Scapula
fossae ( subscapular , supraspinatous , infraspinatous ) notches ( suprascapular , great scapular ) glenoid cavity tubercles ( infraglenoid , supraglenoid )
spine of scapula acromion coracoid process angles ( superior , inferior , lateral ) Humerus upper extremity : necks ( anatomical , surgical ) tubercles (
greater , lesser ) bicipital groove body : radial sulcus deltoid tuberosity lower extremity : capitulum trochlea epicondyles ( lateral , medial ) supracondylar
ridges ( lateral , medial ) fossae ( radial , coronoid , olecranon ) Forearm Radius near elbow ( head , tuberosity ) near wrist ( ulnar notch , styloid process
,
</item 4>

<item 5>
Glenoid cavity
Strong glenohumeral ligaments and muscles prevents dislocation in most cases . By being so shallow the glenoid cavity allows the shoulder joint to have the
greatest mobility of all joints in the body , allowing 120 degrees of unassisted flexion . Additional range of motion in shoulder flexion ( typically up to 180
degrees in humans ) is also accomplished by the great mobility of the scapula ( shoulder blade ) through a process known as scapulohumeral rhythm . Contents (
hide ) 1 Evolution 1.1 In dinosaurs 2 Additional images 3 See also 4 Notes 5 References 6 References 7 External links Evolution ( edit ) Interpretations of the
fossil remains of Australopithecus africanas ( STS 7 ) and A. afarensis ( AL 288 - 1 ; a.k.a. Lucy ) suggest that the glenoid fossa was oriented more cranially
in these species than in modern humans .
</item 5>

<item 6>
Glenoid cavity
ISBN 1 - 4051 - 3413 - 5 . This article incorporates text in the public domain from the 20th edition of Gray 's Anatomy ( 1918 ) References ( edit ) Larson ,
Susan G. ( 2009 ) . `` Evolution of the Hominin Shoulder : Early Homo '' . In Grine , Frederick E. ; Fleagle , John G. ; Leakey , Richard E. The First Humans --
Origin and Early Evolution of the Genus Homo . Springer . doi : 10.1007 / 978 - 1 - 4020 - 9980 - 9 . ISBN 978 - 1 - 4020 - 9979 - 3 . ANATOMY & PHYSIOLOGY :
THE UNITY OF FORM AND FUNCTION , SIXTH EDITION Published by McGraw - Hill Written by Kenneth Saladin
http://www.ucsfhealth.org/conditions/glenoid_labrum_tear/index.html External links ( edit ) Wikimedia Commons has media related to Glenoid cavity . Diagram at
cerrocoso.edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online ,
</item 6>

<item 7>
Glenoid cavity
tuberosity ) near wrist ( ulnar notch , styloid process , Lister 's tubercle ) Ulna near elbow ( tuberosity , olecranon , coronoid process , radial notch ,
trochlear notch ) near wrist ( styloid process ) Hand Carpal bones scaphoid lunate triquetral pisiform trapezium trapezoid capitate hamate hamulus Metacarpal
bones 1st 2nd 3rd 4th 5th Phalanges proximal intermediate distal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Glenoid_cavity&oldid=805938319 ''
Categories : Wikipedia articles incorporating text from the 20th edition of Gray 's Anatomy ( 1918 ) Shoulder Scapula Hidden categories : Articles containing
Greek - language text Medicine infobox template using GraySubject or GrayPage Medicine infobox template using Dorlands parameter Talk Contents About Wikipedia
Deutsch Español فارسی Français Galego Ido Italiano Magyar Português Svenska ไทย Українська Edit links This page was last edited on 18 October 2017 ,
</item 7>

<item 8>
Graph query row:
structure: glenoid cavity
relationship: PART_OF
related: scapula
</item 8>

<item 9>
Glenoid cavity
Animation of glenohumeral joint . The muscle shown is supraspinatus muscle . See also ( edit ) Glenopolar angle Notes ( edit ) Jump up ^ The word glenoid is
pronounced / ˈɡliːnɔɪd / or / ˈɡlɛnɔɪd / ( both are common ) and is from Greek : gléne , `` socket '' , reflecting the shoulder joint 's ball - and - socket
form . References ( edit ) Jump up ^ `` Scapulohumeral Rhythm - Physiopedia , universal access to physiotherapy knowledge '' . www.physio-pedia.com . Retrieved
2016 - 09 - 01 . Jump up ^ Larson 2009 , p. 65 Jump up ^ Martin , A.J. ( 2006 ) . Introduction to the Study of Dinosaurs . Second Edition . Oxford , Blackwell
Publishing . pg. 299 - 300 . ISBN 1 - 4051 - 3413 - 5 .
</item 9>

<item 10>
Glenoid cavity
This reflects the importance of overhead limb postures and suggests a retention of arboreal adaptations in these hominoid primates , whereas the lateral
orientation of the glenoid in modern humans reflects the typical lowered position of the arm . In dinosaurs ( edit ) In dinosaurs the main bones of the pectoral
girdle were the scapula ( shoulder blade ) and the coracoid , both of which directly articulated with the clavicle . The place on the scapula where it
articulated with the humerus ( upper bone of the forelimb ) is called the glenoid . The glenoid is important because it defines the range of motion of the
humerus . Additional images ( edit ) Left scapula . Glenoid cavity shown in red . Animation . Glenoid cavity shown in red . Glenoid fossa of right side .
Anterior view of left scapula showing glenoid cavity `` 2 '' Animation of shoulder joint . The muscles shown are subscapularis muscle ( at right ) ,
infraspinatus muscle ( at top left ) , teres minor muscle ( at bottom left ) .
</item 10>
=== END ITEM id=e23bb75b7bca1b45 ===

=== ITEM id=3d4f6097f079a2d4 ===
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
Independence ( probability theory ) - wikipedia Independence ( probability theory ) Jump to : navigation , search Part of a series on Statistics Probability
theory Probability axioms Probability space Sample space Elementary event Event Random variable Probability measure Complementary event Joint probability
Marginal probability Conditional probability Independence Conditional independence Law of total probability Law of large numbers Bayes ' theorem Boole 's
inequality Venn diagram Tree diagram In probability theory , two events are independent , statistically independent , or stochastically independent if the
occurrence of one does not affect the probability of occurrence of the other . Similarly , two random variables are independent if the realization of one does
not affect the probability distribution of the other . The concept of independence extends to dealing with collections of more than two events or random
variables , in which case the events are pairwise independent if each pair are independent of each other , and the events are mutually independent if each event
is independent of each other combination of events . Contents ( hide ) 1 Definition 1.1 For events 1.1. 1 Two events 1.1. 2 More than two events 1.2 For random
variables 1.2. 1 Two random variables 1.2.
</item 1>

<item 2>
Independence (probability theory)
Thus , the occurrence of B does not affect the probability of A , and vice versa . Although the derived expressions may seem more intuitive , they are not the
preferred definition , as the conditional probabilities may be undefined if P ( A ) or P ( B ) are 0 . Furthermore , the preferred definition makes clear by
symmetry that when A is independent of B , B is also independent of A . More than two events ( edit ) A finite set of events ( A i ) i = 1 n ( \ displaystyle \
( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) is pairwise independent if every pair of events is independent -- that is , if and only if for all distinct pairs of
indices m , k , P ( A m ∩ A k ) = P ( A m ) P ( A k ) . ( \ displaystyle \ mathrm ( P ) ( A_ ( m ) \ cap A_ ( k ) ) = \ mathrm ( P ) ( A_ ( m ) ) \ mathrm ( P )
( A_ ( k ) ) . )
</item 2>

<item 3>
Independence (probability theory)
) and yet no two of the three events are pairwise independent ( and hence the set of events are not mutually independent ) . This example shows that mutual
independence involves requirements on the products of probabilities of all combinations of events , not just the single events as in this example . See also (
edit ) Copula ( statistics ) Independent and identically distributed random variables Mutually exclusive events Subindependence Linear dependence between random
variables Conditional independence Normally distributed and uncorrelated does not imply independent Mean dependence References ( edit ) Jump up ^ Russell ,
Stuart ; Norvig , Peter ( 2002 ) . Artificial Intelligence : A Modern Approach . Prentice Hall . p. 478 . ISBN 0 - 13 - 790395 - 2 . ^ Jump up to : Feller , W (
1971 ) . `` Stochastic Independence '' . An Introduction to Probability Theory and Its Applications . Wiley . Jump up ^ George , Glyn , `` Testing for the
independence of three events , '' Mathematical Gazette 88 , November 2004 , 568 .
</item 3>

<item 4>
Independence (probability theory)
A finite set of events is mutually independent if every event is independent of any intersection of the other events -- that is , if and only if for every k -
element subset of ( A i ) i = 1 n ( \ displaystyle \ ( A_ ( i ) \ ) _ ( i = 1 ) ^ ( n ) ) , P ( ⋂ i = 1 k A i ) = ∏ i = 1 k P ( A i ) . ( \ displaystyle \
mathrm ( P ) \ left ( \ bigcap _ ( i = 1 ) ^ ( k ) A_ ( i ) \ right ) = \ prod _ ( i = 1 ) ^ ( k ) \ mathrm ( P ) ( A_ ( i ) ) . ) This is called the
multiplication rule for independent events . Note that it is not a single condition involving only the product of all the probabilities of all single events (
see below for a counterexample ) ; it must hold true for all subsets of events .
</item 4>

<item 5>
Independence (probability theory)
More than two random variables ( edit ) A set of random variables is pairwise independent if and only if every pair of random variables is independent . Even if
the set of random variables is pairwise independent , it is not necessarily mutually independent as defined next . A set of random variables is mutually
independent if and only if for any finite subset X 1 , ... , X n ( \ displaystyle X_ ( 1 ) , \ ldots , X_ ( n ) ) and any finite sequence of numbers a 1 , ... ,
a n ( \ displaystyle a_ ( 1 ) , \ ldots , a_ ( n ) ) , the events ( X 1 ≤ a 1 ) , ... , ( X n ≤ a n ) ( \ displaystyle \ ( X_ ( 1 ) \ leq a_ ( 1 ) \ ) , \ ldots
, \ ( X_ ( n ) \ leq a_ ( n ) \ ) ) are mutually independent events ( as defined above ) . The measure - theoretically inclined may prefer to substitute events
( X ∈ A ) for events ( X ≤ a ) in the above definition , where A is any Borel set .
</item 5>

<item 6>
Independence (probability theory)
That definition is exactly equivalent to the one above when the values of the random variables are real numbers . It has the advantage of working also for
complex - valued random variables or for random variables taking values in any measurable space ( which includes topological spaces endowed by appropriate σ -
algebras ) . Conditional independence ( edit ) Main article : Conditional independence Intuitively , two random variables X and Y are conditionally independent
given Z if , once Z is known , the value of Y does not add any additional information about X . For instance , two measurements X and Y of the same underlying
quantity Z are not independent , but they are conditionally independent given Z ( unless the errors in the two measurements are somehow connected ) . The formal
definition of conditional independence is based on the idea of conditional distributions .
</item 6>

<item 7>
Independence (probability theory)
The random variables in the first space are pairwise independent because P ( A B ) = P ( A C ) = 1 / 2 = P ( A ) , P ( B A ) = P ( B C ) = 1 / 2 = P ( B ) , and
P ( C A ) = P ( C B ) = 1 / 4 = P ( C ) ; but the three random variables are not mutually independent . The random variables in the second space are both
pairwise independent and mutually independent . To illustrate the difference , consider conditioning on two events . In the pairwise independent case , although
any one event is independent of each of the other two individually , it is not independent of the intersection of the other two : P ( A B C ) = 4 40 4 40 + 1 40
= 4 5 ≠ P ( A ) ( \ displaystyle \ mathrm ( P ) ( A BC ) = ( \ frac ( \ frac ( 4 ) ( 40 ) ) ( ( \ frac ( 4 ) ( 40 ) ) + ( \ frac ( 1 ) ( 40 ) ) ) ) = ( \ tfrac
( 4 )
</item 7>

<item 8>
Independence (probability theory)
For more than two events , a mutually independent set of events is ( by definition ) pairwise independent ; but the converse is not necessarily true ( see below
for a counterexample ) . For random variables ( edit ) Two random variables ( edit ) Two random variables X and Y are independent if and only if ( iff ) the
elements of the π - system generated by them are independent ; that is to say , for every a and b , the events ( X ≤ a ) and ( Y ≤ b ) are independent events (
as defined above ) .
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
A ( \ displaystyle ( \ mathcal ( A ) ) ) and B ( \ displaystyle ( \ mathcal ( B ) ) ) are said to be independent if , whenever A ∈ A ( \ displaystyle A \ in ( \
mathcal ( A ) ) ) and B ∈ B ( \ displaystyle B \ in ( \ mathcal ( B ) ) ) , P ( A ∩ B ) = P ( A ) P ( B ) . ( \ displaystyle \ mathrm ( P ) ( A \ cap B ) = \
mathrm ( P ) ( A ) \ mathrm ( P ) ( B ) . )
</item 10>
=== END ITEM id=3d4f6097f079a2d4 ===

=== ITEM id=143bcc59e95ea40e ===
Gold sentences and facts:
Sentence 0: A defensive three - second violation , also known as illegal defense , is a basketball rules infraction in the National Basketball Association ( NBA ) introduced in the 2001 -- 2002 season . It is assessed when a member of the defending team spends more than three seconds in the free throw lane ( also known as the 16 - foot lane , or colloquially as `` in the paint '' ) while not actively guarding an opponent . To be considered actively guarding , a defender must be within arm 's length of an opponent and in a guarding position . A three - second count is suspended if :
  Fact 0.0: A defensive three-second violation is also known as illegal defense.
  Fact 0.1: A defensive three-second violation is a basketball rules infraction in the National Basketball Association (NBA).
  Fact 0.2: The defensive three-second rule was introduced in the 2001-2002 season.
  Fact 0.3: A defensive three-second violation is assessed when a member of the defending team spends more than three seconds in the free throw lane while not actively guarding an opponent.
  Fact 0.4: The free throw lane is also known as the 16-foot lane.
  Fact 0.5: The free throw lane is colloquially known as 'in the paint'.
  Fact 0.6: To be considered actively guarding, a defender must be within arm's length of an opponent.
  Fact 0.7: To be considered actively guarding, a defender must be in a guarding position.
  Fact 0.8: A three-second count is suspended under conditions listed after the word 'if'.

Retrieved items:
<item 1>
Defensive three-second violation
corners Motion Princeton Run and gun Grinnell System Nellie Ball Small ball Triangle Defense 1 - 2 - 1 - 1 1 -- 3 -- 1 2 -- 3 zone Amoeba Box - and - one Double
team Full - court press Hack - a-Shaq Jordan Rules Line Man - to - man Match - up zone Triangle - and - two Zone Equipment Backboard Ball Breakaway rim Court
Half court Key Net Possession arrow Whistle Clothing Basketball sleeve Finger sleeve Air Jordan Chuck Taylor All - Stars Miscellaneous 100 - point scorers 50 --
40 -- 90 club AAU All - Americans AAU champions Mikan Drill Player tracking Shootaround Tip drill Winning streaks This basketball - related article is a stub .
You can help Wikipedia by expanding it . Retrieved from `` https://en.wikipedia.org/w/index.php?title=Defensive_three-second_violation&oldid=803130800 ''
Categories : Basketball penalties Basketball stubs Hidden categories : Wikipedia articles needing clarification from November 2015 All stub articles Talk About
Wikipedia Français Edit links This page was last edited on 30 September 2017 , at 17 : 04 . About Wikipedia
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
NBC Sunday Night Football introduced permanent score displays in 2006, replacing periodic score updates and establishing graphics later adopted across NBC
Sports, with variations by sport. The 2009 redesign added team-colored down-and-distance details and animated touchdown announcements; 2010 brought timeout
indicators and other adjustments. A unified 2012 package improved readability and used team colors. In 2014, NBC added a virtual line of scrimmage and moved the
play clock, then adopted 16:9 letterboxing. A new
</item 3>

<item 4>
Defensive three-second violation
Defensive three - second violation - Wikipedia Defensive three - second violation Jump to : navigation , search A defensive three - second violation , also
known as illegal defense , is a basketball rules infraction in the National Basketball Association ( NBA ) introduced in the 2001 -- 2002 season . It is
assessed when a member of the defending team spends more than three seconds in the free throw lane ( also known as the 16 - foot lane , or colloquially as `` in
the paint '' ) while not actively guarding an opponent . To be considered actively guarding , a defender must be within arm 's length of an opponent and in a
guarding position . A three - second count is suspended if : a player is in the act of shooting there is a loss of team control the defender is actively
guarding an opponent the defender clears the painted area it is imminent the defender will become legal In addition , a player guarding an opponent with the
ball may be in the paint without actively guarding the opponent . The team committing a defensive three - second violation is assessed a team technical foul .
The offense receives one free throw and retains possession of the ball .
</item 4>

<item 5>
Defensive three-second violation
`` PBA get fresh look this season '' . Arab News . Retrieved January 13 , 2014 . Jump up ^ `` WNBA Announces Rules Changes for 2013 Season '' . WNBA . Retrieved
Dec 16 , 2012 . Basketball ( outline ) General topics History of basketball James Naismith Variations of basketball Leagues Statistics Glossary of terms Index
of articles Rules Bonus Jump ball Officials Turnover Violation 3 seconds Offense Defense 5 seconds Basket interference Carrying Double dribble Goaltending Shot
clock Traveling Foul Flagrant Personal Technical Game play Air ball Alley - oop Assist Backboard shattering Ball hog Block Buzzer beater Cherry picking Dribble
Crossover Dunk Euro step Fadeaway Fast break Fly Flop Jump shot Layup Finger roll Field goal Four - point play Free throw Hook shot Moves Pick and roll
Positions Posterized Playbook Rebound Point Screen Back screen Slashing Steal Three - pointer Three - point play Uncontested shot Strategy General Sixth man
Offense Continuity offense Flex Shuffle UCLA High Post Wheel Dribble drive motion Four corners Motion Princeton Run and gun Grinnell System Nellie Ball Small
ball Triangle Defense 1
</item 5>

<item 6>
A defensive three-second violation occurs in the NBA when a defender stays in the free-throw lane for over three seconds without actively guarding an opponent
(generally within arm’s length and in a guarding position). The count pauses during a shot, loss of team control, active guarding, when the defender leaves the
lane, or when the defender is about to become legal. A violation is a team technical: the offense gets one free throw and keeps possession. Introduced in
2001–02, the rule
</item 6>

<item 7>
Defensive three-second violation
The WNBA started using it in 2013 . References ( edit ) Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , B. Retrieved July 25 , 2010 Jump up
^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , D. Retrieved July 25 , 2010 Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , E.
Retrieved July 25 , 2010 Jump up ^ NBA Official Rules ( 2009 - 2010 ) Rule 10 , Section VII , Penalty . Retrieved July 25 , 2010 Jump up ^ Blinebury , Fran (
June 8 , 2006 ) . `` Rule changes have NBA back in the fast lane '' . Houston Chronicle . Retrieved July 26 , 2010 . Jump up ^
http://www.sportingcharts.com/dictionary/nba/defensive-three-second-violation.aspx Jump up ^ Cruz , Agnes ( January 24 , 2003 ) . `` PBA get fresh look this
season '' . Arab News .
</item 7>

<item 8>
The 2017–18 NBA season began eight days earlier to reduce back-to-back games and discourage resting stars. Rule changes cut timeouts from 18 to 14, moved the
trade deadline to before the All-Star break, penalized free-throw shooters who crossed the three-point line between attempts, and introduced two-way contracts.
Officials gained authority to penalize dangerous landing-space closeouts (“Zaza rule”) and foul-drawing tactics such as lunging into defenders or locking arms
(“Harden rule
</item 8>

<item 9>
Defensive three-second violation
The offense receives one free throw and retains possession of the ball . Prior to the 2001 -- 2002 NBA season , any form of zone defense was considered an
illegal defense violation and resulted in a warning on the first violation and then a technical foul for any subsequent violations . However , the defensive
three - second violation makes it difficult for NBA defenses to play zone , since zone defenses usually position a player in the middle of the key to stop
penetration . The Philippine Basketball Association used to follow the illegal defense rule until the 2003 PBA season when it was abolished . The high school
game does not use this rule , nor does European basketball . The WNBA started using it in 2013 .
</item 9>

<item 10>
2017–18 NBA season
Informally dubbed as the `` Zaza Pachulia rule '' . Also , officials can now assess if a player is in a shooting motion when determining if a perimeter foul is
a shooting or a common foul . The type of foul will be determined on the sequencing of the player 's movement if the foul happens before or after the player
starts a shooting motion . Additionally , they can also assess an offensive foul if an offensive player intentionally lunges into the defender or locks arms to
create contact , a rule informally dubbed as the `` Harden rule '' . On September 28 , 2017 , the NBA board of governors approved the changes to the draft
lottery system to discourage teams from losing on purpose . Starting with the 2018 -- 19 NBA season , the teams with the three worst records will have equal
odds at the # 1 pick . In addition , the revised draft lottery will consist of selecting the top four teams instead of the usual three that occurred for over
two decades . The approval for the change was nearly unanimous , with only the Oklahoma City Thunder voting against the change and the Dallas Mavericks
abstaining from voting .
</item 10>
=== END ITEM id=143bcc59e95ea40e ===

=== ITEM id=b75b0a3487030676 ===
Gold sentences and facts:
Sentence 0: Take action immediately upon hearing thunder . Anyone close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid touching metal .
  Fact 0.0: Take action immediately upon hearing thunder.
  Fact 0.1: Anyone close enough to the storm to hear thunder can be struck by lightning.
  Fact 0.2: Avoid electrical appliances, including corded telephones.
  Fact 0.3: Cordless and wireless telephones are safe to use during a thunderstorm.
  Fact 0.4: Close and stay away from windows and doors.
  Fact 0.5: Glass can become a serious hazard in high wind.
  Fact 0.6: Do not bathe or shower.
  Fact 0.7: Plumbing conducts electricity.
  Fact 0.8: If driving, safely exit the roadway.
  Fact 0.9: If driving, turn on hazard lights.
  Fact 0.10: If driving, park.
  Fact 0.11: Remain in the vehicle and avoid touching metal.
Sentence 1: Take action immediately upon hearing thunder . Anyone close enough to the storm to hear thunder can be struck by lightning .
  Fact 1.0: Take action immediately upon hearing thunder.
  Fact 1.1: Anyone close enough to the storm to hear thunder can be struck by lightning.

Retrieved items:
<item 1>
Thunderstorm
Safety ( edit ) While safety and preparedness often overlap , `` thunderstorm safety '' generally refers to what people should do during and after a storm . The
American Red Cross recommends that people follow these precautions if a storm is imminent or in progress : Take action immediately upon hearing thunder . Anyone
close enough to the storm to hear thunder can be struck by lightning . Avoid electrical appliances , including corded telephones . Cordless and wireless
telephones are safe to use during a thunderstorm . Close and stay away from windows and doors , as glass can become a serious hazard in high wind . Do not bathe
or shower , as plumbing conducts electricity . If driving , safely exit the roadway , turn on hazard lights , and park . Remain in the vehicle and avoid
touching metal . The NWS stopped recommending the `` lightning crouch '' in 2008 as it does n't provide a significant level of protection and will not
significantly lower the risk of being killed or injured from a nearby lightning strike . Frequent occurrences ( edit ) See also : United States rainfall
climatology A mild thunderstorm over Niagara Falls , Ontario .
</item 1>

<item 2>
Thunderstorm
Radio pulses produced by cosmic rays are being used to study how electric charges develop within thunderstorms . More organized meteorological projects such as
VORTEX2 use an array of sensors , such as the Doppler on Wheels , vehicles with mounted automated weather stations , weather balloons , and unmanned aircraft to
investigate thunderstorms expected to produce severe weather . Lightning is detected remotely using sensors that detect cloud - to - ground lightning strokes
with 95 percent accuracy in detection and within 250 metres ( 820 ft ) of their point of origin . Mythology and religion ( edit ) Thunderstorms strongly
influenced many early civilizations . Greeks believed that they were battles waged by Zeus , who hurled lightning bolts forged by Hephaestus . Some American
Indian tribes associated thunderstorms with the Thunderbird , who they believed was a servant of the Great Spirit . The Norse considered thunderstorms to occur
when Thor went to fight Jötnar , with the thunder and lightning being the effect of his strikes with the hammer Mjölnir . Hinduism recognizes Indra as the god
of rain and thunderstorms .
</item 2>

<item 3>
Thunderstorm
Motion ( edit ) Thunderstorm line viewed in reflectivity ( dBZ ) on a plan position indicator radar display The two major ways thunderstorms move are via
advection of the wind and propagation along outflow boundaries towards sources of greater heat and moisture . Many thunderstorms move with the mean wind speed
through the Earth 's troposphere , the lowest 8 kilometres ( 5.0 mi ) of the Earth 's atmosphere . Weaker thunderstorms are steered by winds closer to the Earth
's surface than stronger thunderstorms , as the weaker thunderstorms are not as tall . Organized , long - lived thunderstorm cells and complexes move at a right
angle to the direction of the vertical wind shear vector . If the gust front , or leading edge of the outflow boundary , races ahead of the thunderstorm , its
motion will accelerate in tandem . This is more of a factor with thunderstorms with heavy precipitation ( HP ) than with thunderstorms with low precipitation (
LP ) . When thunderstorms merge , which is most likely when numerous thunderstorms exist in proximity to each other , the motion of the stronger thunderstorm
normally dictates the future motion of the merged cell .
</item 3>

<item 4>
Thunderstorm
doi : 10.1111 / cea. 12709 . PMID 26765082 ^ Jump up to : American Red Cross . `` Thunderstorm Safety Checklist '' ( PDF ) . American Red Cross . Retrieved 24
August 2011 . ^ Jump up to : National Weather Service Weather Forecast Office . `` Thunderstorm '' . Severe Weather Preparedness Information . Albuquerque , NM
: NOAA . Retrieved 24 August 2011 . Jump up ^ Federal Emergency Management Agency . `` Thunderstorms and Lightning '' . Ready . US Department of Homeland
Security . Archived from the original on 23 June 2011 . Retrieved 24 August 2011 . ^ Jump up to : Federal Emergency Management Agency . `` What to Do Before a
Thunderstorm '' . US Department of Homeland Security . Archived from the original on 20 August 2011 . Retrieved 24 August 2011 . ^ Jump up to : `` NWS Lightning
Safety Myths '' . Lightningsafety.noaa.gov. 2014 - 06 - 30 .
</item 4>

<item 5>
Thunderstorm
A similar event occurred in Boscastle , England , on 16 August 2004 . Hazards ( edit ) Each year , many people are killed or seriously injured by severe
thunderstorms despite the advance warning . While severe thunderstorms are most common in the spring and summer , they can occur at just about any time of the
year . Cloud - to - ground lightning ( edit ) See also : Lightning strike and Wildfire A return stroke , cloud - to - ground lightning strike during a
thunderstorm . Cloud - to - ground lightning frequently occurs within the phenomena of thunderstorms and have numerous hazards towards landscapes and
populations . One of the more significant hazards lightning can pose is the wildfires they are capable of igniting . Under a regime of low precipitation ( LP )
thunderstorms , where little precipitation is present , rainfall can not prevent fires from starting when vegetation is dry as lightning produces a concentrated
amount of extreme heat . Direct damage caused by lightning strikes occurs on occasion . In areas with a high frequency for cloud - to - ground lightning , like
Florida , lightning causes several fatalities per year , most commonly to people working outside . Acid rain is also a frequent risk produced by lightning .
</item 5>

<item 6>
Thunderstorm
Supercell thunderstorms are the strongest and most severe . Mesoscale convective systems formed by favorable vertical wind shear within the tropics and
subtropics can be responsible for the development of hurricanes . Dry thunderstorms , with no precipitation , can cause the outbreak of wildfires from the heat
generated from the cloud - to - ground lightning that accompanies them . Several means are used to study thunderstorms : weather radar , weather stations , and
video photography . Past civilizations held various myths concerning thunderstorms and their development as late as the 18th century . Beyond the Earth 's
atmosphere , thunderstorms have also been observed on the planets of Jupiter , Saturn , Neptune , and , probably , Venus .
</item 6>

<item 7>
Thunderstorm
Mature stage ( edit ) Anvil - shaped thundercloud in the mature stage over Swifts Creek , Victoria In the mature stage of a thunderstorm , the warmed air
continues to rise until it reaches an area of warmer air and can rise no farther . Often this ' cap ' is the tropopause . The air is instead forced to spread
out , giving the storm a characteristic anvil shape . The resulting cloud is called cumulonimbus incus . The water droplets coalesce into larger and heavier
droplets and freeze to become ice particles . As these fall , they melt to become rain . If the updraft is strong enough , the droplets are held aloft long
enough to become so large that they do not melt completely but fall as hail . While updrafts are still present , the falling rain drags the surrounding air with
it , creating downdrafts as well . The simultaneous presence of both an updraft and a downdraft marks the mature stage of the storm and produces cumulonimbus
clouds . During this stage , considerable internal turbulence can occur within , which manifests as strong winds , severe lightning , and even tornadoes .
</item 7>

<item 8>
Thunderstorm
Retrieved 2009 - 09 - 09 . Jump up ^ Charles H. Paxton ; J. Colson & N. Carlisle ( 2008 ) . `` P2. 13 Florida lightning deaths and injuries 2004 -- 2007 '' .
American Meteorological Society . Retrieved 2009 - 09 - 05 . Jump up ^ G.E. Likens ; W.C. Keene ; J.M. Miller & J.N. Galloway ( 1987 ) . `` Chemistry of
precipitation from a remote , terrestrial site in Australia '' . Journal of Geophysical Research . 92 ( 13 ) : 299 -- 314 . Bibcode : 1987JGR ... 92 ... 299R .
doi : 10.1029 / JA092iA01p00299 . Jump up ^ Joel S. Levine ; Tommy R. Augustsson ; Iris C. Andersont ; James M. Hoell Jr . & Dana A. Brewer ( 1984 ) . ``
Tropospheric sources of NOx : Lightning and biology '' . Atmospheric Environment.
</item 8>

<item 9>
Thunderstorm
Lightningsafety.noaa.gov. 2014 - 06 - 30 . Retrieved 2014 - 08 - 20 . Jump up ^ `` NWS JetStream -- Lightning Frequently Asked Questions '' . Srh.noaa.gov. 2014
- 06 - 28 . Retrieved 2014 - 08 - 20 . Jump up ^ `` You 're not safer crouching : Six things you did n't know about lightning '' . LA Times . Retrieved 2014 -
08 - 20 . Jump up ^ National Geographic Almanac of Geography , ISBN 0 - 7922 - 3877 - X , page 75 . Jump up ^ `` How many thunderstorms occur each year ? '' .
Thunderstorms . Sky Fire Productions . Archived from the original on 2007 - 07 - 11 . Retrieved 2006 - 08 - 23 . Jump up ^ National Weather Service JetStream (
2008 - 10 - 08 ) . `` Tropical Cyclone Hazards '' . National Weather Service Southern Region Headquarters .
</item 9>

<item 10>
Thunderstorm
) Fog Convective condensation level ( CCL ) Lifted condensation level ( LCL ) Precipitation Water vapor Convection Convective available potential energy ( CAPE
) Convective inhibition ( CIN ) Convective instability Convective momentum transport Convective temperature ( T ) Equilibrium level ( EL ) Free convective layer
( FCL ) Helicity K Index Level of free convection ( LFC ) Lifted index ( LI ) Maximum parcel level ( MPL ) Bulk Richardson number ( BRN ) Temperature Dew point
( T ) Dew point depression Dry - bulb temperature Equivalent temperature ( T ) Forest fire weather index Haines Index Heat index Humidex Humidity Relative
humidity ( RH ) Mixing ratio Potential temperature ( θ ) Equivalent potential temperature ( θ ) Sea surface temperature ( SST ) Thermodynamic temperature Vapor
pressure Virtual temperature Wet - bulb temperature Wet - bulb potential temperature Wind chill Pressure Atmospheric pressure Baroclinity Barotropicity Pressure
gradient Pressure - gradient force ( PGF ) Retrieved from `` https://en.wikipedia.org/w/index.php?title=Thunderstorm&oldid=821769562 '' Categories : Lightning
Atmospheric electricity Weather hazards to aircraft Microscale meteorology Severe weather and convection Storm Weather hazards Rain Hidden categories :
</item 10>
=== END ITEM id=b75b0a3487030676 ===

=== ITEM id=6e9d3f24c9d3c27f ===
Gold sentences and facts:
Sentence 0: The ability to compare ammunition is a direct result of the invention of rifling around the turn of the 16th century . By forcing the bullet to spin as it travels down the barrel of the weapon the bullet 's accuracy greatly increases . At the same time , the rifling leaves marks on the bullet that are indicative of that particular barrel . Prior to mass production of firearms , each barrel and bullet mold was hand made by gunsmiths making them unique . The first successful documented case of forensic firearm examination occurred in 1835 when a member of the Bow Street Runners in London matched a recovered bullet from a murder victim to a specific mold in a suspect 's home confirming that he made the bullet . Further evidence that the bullet maker was the perpetrator was found in his home and he was convicted . As manufacturing and automation replaced hand tools , the ability to compare bullets became impossible due to the standardization of molds within a specific company . However , experts in the field postulated that there were microscopic differences on each barrel left during the manufacturing process . These differences were a result of wear on the machines and since each new weapon caused a tiny amount of wear , each barrel would be slightly different from every other barrel produced by that company . Also , each bullet fired from a specific barrel would be printed with the same marks , allowing investigators to identify the weapon that fired a specific bullet .
  Fact 0.0: The ability to compare ammunition is a direct result of the invention of rifling.
  Fact 0.1: Rifling was invented around the turn of the 16th century.
  Fact 0.2: Rifling forces the bullet to spin as it travels down the barrel of the weapon.
  Fact 0.3: Rifling greatly increases the bullet's accuracy.
  Fact 0.4: At the same time, rifling leaves marks on the bullet that are indicative of that particular barrel.
  Fact 0.5: Before mass production of firearms, each barrel and bullet mold was hand made by gunsmiths, making each one unique.
  Fact 0.6: The first successful documented case of forensic firearm examination occurred in 1835.
  Fact 0.7: In that 1835 case, a member of the Bow Street Runners in London matched a recovered bullet from a murder victim to a specific mold in a suspect's home.
  Fact 0.8: In that 1835 case, the match confirmed that the suspect made the bullet.
  Fact 0.9: Further evidence that the bullet maker was the perpetrator was found in his home.
  Fact 0.10: The bullet maker was convicted.
  Fact 0.11: As manufacturing and automation replaced hand tools, the ability to compare bullets became impossible.
  Fact 0.12: The ability to compare bullets became impossible due to the standardization of molds within a specific company.
  Fact 0.13: Experts in the field postulated that there were microscopic differences on each barrel left during the manufacturing process.
  Fact 0.14: These differences resulted from wear on the machines.
  Fact 0.15: Since each new weapon caused a tiny amount of wear, each barrel would be slightly different from every other barrel produced by that company.
  Fact 0.16: Each bullet fired from a specific barrel would be printed with the same marks.
  Fact 0.17: The same marks allow investigators to identify the weapon that fired a specific bullet.

Retrieved items:
<item 1>
Forensic firearm examination
The invention of the comparison microscope by Calvin Goddard and Phillip O. Gravelle in 1925 modernized the forensic examination of firearms . Simultaneous
comparison of two different objects at the same time allowed to closely examine striations for matches and therefore make a more definitive statement as to
whether or not they matched . One of the first true tests of this new technology was in the aftermath of the Saint Valentine 's Day Massacre in 1929 . During
the Prohibition Era , competing gang members were fighting over bootlegging operations within the city of Chicago . Members of the Chicago Outfit and the Egan
's Rats led by Al Capone attempted to remove all competition from Chicago by eliminating the North Side Gang leader Bugs Moran . The massacre missed Moran , who
was not present , but killed seven members of the North Side Gang . The murderers attempted to cover up their crime by posing as police officers , even dressing
in police uniforms . Witnesses saw two `` officers '' leaving the scene , which implicated the Chicago police department as the perpetrators of the massacre .
High levels of police corruption during that time period made it seem likely that the police department committed the killings .
</item 1>

<item 2>
Forensic firearm examination
Further evidence that the bullet maker was the perpetrator was found in his home and he was convicted . As manufacturing and automation replaced hand tools ,
the ability to compare bullets became impossible due to the standardization of molds within a specific company . However , experts in the field postulated that
there were microscopic differences on each barrel left during the manufacturing process . These differences were a result of wear on the machines and since each
new weapon caused a tiny amount of wear , each barrel would be slightly different from every other barrel produced by that company . Also , each bullet fired
from a specific barrel would be printed with the same marks , allowing investigators to identify the weapon that fired a specific bullet . One of the first uses
of this knowledge was in 1915 to exonerate Charles Stielow of the murder of his neighbors . Stielow was sentenced to death and appealed to Charles S. Whitman ,
the Governor of New York , who was not convinced by the evidence used to convict Stielow . Whitman halted the execution until an inquiry could be conducted and
after further examination it was shown that Stielow 's firearm could not have fired the bullets recovered from the victims .
</item 2>

<item 3>
DNA
However , identification can be complicated if the scene is contaminated with DNA from several people . DNA profiling was developed in 1984 by British
geneticist Sir Alec Jeffreys , and first used in forensic science to convict Colin Pitchfork in the 1988 Enderby murders case . The development of forensic
science and the ability to now obtain genetic matching on minute samples of blood , skin , saliva , or hair has led to re-examining many cases . Evidence can
now be uncovered that was scientifically impossible at the time of the original examination . Combined with the removal of the double jeopardy law in some
places , this can allow cases to be reopened where prior trials have failed to produce sufficient evidence to convince a jury . People charged with serious
crimes may be required to provide a sample of DNA for matching purposes . The most obvious defense to DNA matches obtained forensically is to claim that
cross-contamination of evidence has occurred . This has resulted in meticulous strict handling procedures with new cases of serious crime . DNA profiling is
also used successfully to positively identify victims of mass casualty incidents , bodies or body parts in serious accidents , and individual victims in mass
war graves , via matching to family members .
</item 3>

<item 4>
Stone Age
This is a Mode 1 , or Oldowan , stone tool from the western Sahara . Oldowan in Africa Main article : Oldowan The earliest documented stone tools have been
found in eastern Africa , manufacturers unknown , at the 3.3 million year old site of Lomekwi 3 in Kenya . Better known are the later tools belonging to an
industry known as Oldowan , after the type site of Olduvai Gorge in Tanzania . The tools were formed by knocking pieces off a river pebble , or stones like it ,
with a hammerstone to obtain large and small pieces with one or more sharp edges . The original stone is called a core ; the resultant pieces , flakes .
Typically , but not necessarily , small pieces are detached from a larger piece , in which case the larger piece may be called the core and the smaller pieces
the flakes . The prevalent usage , however , is to call all the results flakes , which can be confusing . A split in half is called bipolar flaking .
Consequently , the method is often called `` core - and - flake '' .
</item 4>

<item 5>
Stone Age
Some Oldowan tools are more carefully prepared to form a more regular edge . One distinguishing criterion is the size of the flakes . In contrast to the Oldowan
`` small flake '' tradition , Acheulean is `` large flake : '' `` The primary technological distinction remaining between Oldowan and the Acheulean is the
preference for large flakes ( > 10 cm ) as blanks for making large cutting tools ( handaxes and cleavers ) in the Acheulean . '' `` Large Cutting Tool ( LCT )
'' has become part of the standard terminology as well . In North Africa , the presence of Mode 2 remains a mystery , as the oldest finds are from Thomas Quarry
in Morocco at 0.9 mya . Archaeological attention , however , shifts to the Jordan Rift Valley , an extension of the East African Rift Valley ( the east bank of
the Jordan is slowly sliding northward as East Africa is thrust away from Africa ) . Evidence of use of the Nile Valley is in deficit , but Hominans could
easily have reached the palaeo - Jordan river from Ethiopia along the shores of the Red Sea , one side or the other .
</item 5>

<item 6>
Stone Age
Starting in the grasslands of the rift , Homo erectus , the predecessor of modern humans , found an ecological niche as a tool - maker and developed a
dependence on it , becoming a `` tool equipped savanna dweller . '' Stone Age in archaeology Beginning of the Stone Age Obsidian projectile point The oldest
indirect evidence found of stone tool use is fossilised animal bones with tool marks ; these are 3.4 million years old and were found in the Lower Awash Valley
in Ethiopia . Archaeological discoveries in Kenya in 2015 , identifying possibly the oldest known evidence of hominin use of tools to date , have indicated that
Kenyanthropus platyops ( a 3.2 to 3.5 - million - year - old Pliocene hominin fossil discovered in Lake Turkana , Kenya in 1999 ) may have been the earliest
tool - users known . The oldest stone tools were excavated from the site of Lomekwi 3 in West Turkana , northwestern Kenya , and date to 3.3 million years old .
</item 6>

<item 7>
Forensic firearm examination
Matching striations can be seen . Spent cartridges found at a scene can be examined for physical evidence such as fingerprints or compared to samples that match
them to a weapon . The examination of the cartridge relies on the unique tool marks left by the various parts of the weapon including the firing pin and the
ejector in semi and fully automatic firearms . These markings can be compared and matched to known exemplars fired from the same weapon using the same parts .
The examination of the marks left on the cartridge is done using a comparison microscope . Examiners view the questioned cartridge and the known exemplar
simultaneously , looking for similar microscopic marks left during the firing process . Example of microstamping . Insert shows a close up of the serial number
imprinted into the cartridge . Cartridges are also routinely examined for fingerprints as the act of loading the ammunition into the magazine , or chamber ,
leaves recoverable impressions . These fingerprints can survive the firing processes and , while a rare occurrence , fingerprints have been obtained from
cartridges recovered from the scene . Cartridges are subjected to cyanoacrylate fuming and examined for any usable prints .
</item 7>

<item 8>
Forensic firearm examination
1471 of 2007 . Retrieved on June 18 , 2016 . Jump up ^ Mather , Kate ( January 23 , 2014 ) . `` Smith & Wesson says it wo n't follow California ' microstamping
' law '' . Los Angeles Times . Retrieved June 18 , 2016 . ^ Jump up to : DiMaio , Vincent J.M. ( 2016 ) . Gunshot Wounds : Practical Aspects of Firearms ,
Ballistics , and Forensic Techniques ( 3rd ed . ) . CRC Press . Retrieved June 4 , 2016 . ^ Jump up to : `` Firearms & Tool Mark '' . North Carolina Department
of Justice . Retrieved June 4 , 2016 . ^ Jump up to : `` Firearms and Toolmarks in the FBI Laboratory '' . Forensic Science Communications . 2 ( 2 ) . April
2000 . Archived from the original on September 20 , 2015 . Retrieved June 5 , 2016 . Jump up ^ Fisher , Barry A.J.
</item 8>

<item 9>
Forensic firearm examination
1 Magnetic particle inspection 2.2. 2 Chemical restoration 3 Examination of cartridges 4 Examination of bullets 4.1 Class characteristics 4.2 Individual
characteristics 5 Striation databasing 6 Criticisms 7 See also 8 References History ( edit ) The ability to compare ammunition is a direct result of the
invention of rifling around the turn of the 16th century . By forcing the bullet to spin as it travels down the barrel of the weapon the bullet 's accuracy
greatly increases . At the same time , the rifling leaves marks on the bullet that are indicative of that particular barrel . Prior to mass production of
firearms , each barrel and bullet mold was hand made by gunsmiths making them unique . The first successful documented case of forensic firearm examination
occurred in 1835 when a member of the Bow Street Runners in London matched a recovered bullet from a murder victim to a specific mold in a suspect 's home
confirming that he made the bullet . Further evidence that the bullet maker was the perpetrator was found in his home and he was convicted .
</item 9>

<item 10>
Stone Age
There is additional evidence that Oldowan , or Mode 1 , tools were utilized in `` percussion technology '' ; that is , they were designed to be gripped at the
blunt end and strike something with the edge , from which use they were given the name of choppers . Modern science has been able to detect mammalian blood
cells on Mode 1 tools at Sterkfontein , Member 5 East , in South Africa . As the blood must have come from a fresh kill , the tool users are likely to have done
the killing and used the tools for butchering . Plant residues bonded to the silicon of some tools confirm the use to chop plants . Although the exact species
authoring the tools remains unknown , Mode 1 tools in Africa were manufactured and used predominantly by Homo habilis . They can not be said to have developed
these tools or to have contributed the tradition to technology . They continued a tradition of yet unknown origin . As chimpanzees sometimes naturally use
percussion to extract or prepare food in the wild , and may use either unmodified stones or stones that they have split , creating an Oldowan tool , the
tradition may well be far older than its current record .
</item 10>
=== END ITEM id=6e9d3f24c9d3c27f ===

=== ITEM id=8b347d481d15ef0b ===
Gold sentences and facts:
Sentence 0: Manchester United Football Club is a professional football club based in Old Trafford , Greater Manchester , England , that competes in the Premier League , the top flight of English football . Nicknamed `` the Red Devils '' , the club was founded as Newton Heath LYR Football Club in 1878 , changed its name to Manchester United in 1902 and moved to its current stadium , Old Trafford , in 1910 .
  Fact 0.0: Manchester United Football Club is a professional football club.
  Fact 0.1: Manchester United Football Club is based in Old Trafford, Greater Manchester, England.
  Fact 0.2: Manchester United Football Club competes in the Premier League, the top flight of English football.
  Fact 0.3: Manchester United Football Club is nicknamed "the Red Devils".
  Fact 0.4: Manchester United Football Club was founded as Newton Heath LYR Football Club in 1878.
  Fact 0.5: Manchester United Football Club changed its name to Manchester United in 1902.
  Fact 0.6: Manchester United Football Club moved to its current stadium, Old Trafford, in 1910.

Retrieved items:
<item 1>
Manchester United F.C.
These were dismantled in 1987 and replaced by a lighting system embedded in the roof of each stand , which remains in use today . The Taylor Report 's
requirement for an all - seater stadium lowered capacity at Old Trafford to around 44,000 by 1993 . In 1995 , the North Stand was redeveloped into three tiers ,
restoring capacity to approximately 55,000 . At the end of the 1998 -- 99 season , second tiers were added to the East and West Stands , raising capacity to
around 67,000 , and between July 2005 and May 2006 , 8,000 more seats were added via second tiers in the north - west and north - east quadrants . Part of the
new seating was used for the first time on 26 March 2006 , when an attendance of 69,070 became a new Premier League record .
</item 1>

<item 2>
Manchester United F.C.
Bombing in the Second World War destroyed much of the stadium ; the central tunnel in the South Stand was all that remained of that quarter . After the war ,
the club received compensation from the War Damage Commission in the amount of £ 22,278 . While reconstruction took place , the team played its `` home '' games
at Manchester City 's Maine Road ground ; Manchester United was charged £ 5,000 per year , plus a nominal percentage of gate receipts . Later improvements
included the addition of roofs , first to the Stretford End and then to the North and East Stands . The roofs were supported by pillars that obstructed many
fans ' views , and they were eventually replaced with a cantilevered structure . The Stretford End was the last stand to receive a cantilevered roof , completed
in time for the 1993 -- 94 season . First used on 25 March 1957 and costing £ 40,000 , four 180 - foot ( 55 m ) pylons were erected , each housing 54 individual
floodlights .
</item 2>

<item 3>
Manchester United F.C.
The club has the third highest social media following in the world among sports teams ( after Barcelona and Real Madrid ) , with over 71 million Facebook fans
as of September 2016 . A 2014 study showed that Manchester United had the loudest fans in the Premier League . Supporters are represented by two independent
bodies ; the Independent Manchester United Supporters ' Association ( IMUSA ) , which maintains close links to the club through the MUFC Fans Forum , and the
Manchester United Supporters ' Trust ( MUST ) . After the Glazer family 's takeover in 2005 , a group of fans formed a splinter club , F.C. United of Manchester
. The West Stand of Old Trafford -- the `` Stretford End '' -- is the home end and the traditional source of the club 's most vocal support . Rivalries Main
articles : Arsenal F.C. -- Manchester United F.C. rivalry , Manchester derby , Liverpool F.C. -- Manchester United F.C. rivalry , and Leeds United F.C. --
Manchester United F.C.
</item 3>

<item 4>
Manchester United F.C.
Within four years , the stadium had cover on all four sides , as well as the ability to hold approximately 50,000 spectators , some of whom could watch from the
viewing gallery atop the Main Stand . However , following Manchester United 's first league title in 1908 and the FA Cup a year later , it was decided that Bank
Street was too restrictive for Davies ' ambition ; in February 1909 , six weeks before the club 's first FA Cup title , Old Trafford was named as the home of
Manchester United , following the purchase of land for around £ 60,000 . Architect Archibald Leitch was given a budget of £ 30,000 for construction ; original
plans called for seating capacity of 100,000 , though budget constraints forced a revision to 77,000 . The building was constructed by Messrs Brameld and Smith
of Manchester . The stadium 's record attendance was registered on 25 March 1939 , when an FA Cup semi-final between Wolverhampton Wanderers and Grimsby Town
drew 76,962 spectators .
</item 4>

<item 5>
Manchester United F.C.
Manchester United F.C. - Wikipedia Manchester United F.C. Jump to : navigation , search `` MUFC '' and `` Manchester United '' redirect here . For other uses ,
see MUFC ( disambiguation ) . For the basketball team , see Manchester Eagles . Not to be confused with F.C. United of Manchester . Manchester United Full name
Manchester United Football Club Nickname ( s ) The Red Devils Short name MUFC Founded 1878 ; 140 years ago ( 1878 ) , as Newton Heath LYR F.C. 1902 ; 116 years
ago ( 1902 ) , as Manchester United F.C. Ground Old Trafford Ground Capacity 74,994 Owner Manchester United plc ( NYSE : MANU ) Co-chairmen Joel and Avram
Glazer Manager José Mourinho League Premier League 2016 -- 17 Premier League , 6th of 20 Website Club website Home colours Away colours Third colours Current
season Manchester United Football Club is a professional football club based in Old Trafford , Greater Manchester , England , that competes in the Premier
League , the top flight of English football .
</item 5>

<item 6>
Manchester United F.C.
Some expansion took place in 1887 , and in 1891 , Newton Heath used its minimal financial reserves to purchase two grandstands , each able to hold 1,000
spectators . Although attendances were not recorded for many of the earliest matches at North Road , the highest documented attendance was approximately 15,000
for a First Division match against Sunderland on 4 March 1893 . A similar attendance was also recorded for a friendly match against Gorton Villa on 5 September
1889 . In June 1893 , after the club was evicted from North Road by its owners , Manchester Deans and Canons , who felt it was inappropriate for the club to
charge an entry fee to the ground , secretary A.H. Albut procured the use of the Bank Street ground in Clayton . It initially had no stands , by the start of
the 1893 -- 94 season , two had been built ; one spanning the full length of the pitch on one side and the other behind the goal at the `` Bradford end '' . At
the opposite end , the `` Clayton end '' , the ground had been `` built up , thousands thus being provided for '' .
</item 6>

<item 7>
Manchester United F.C.
Non-League ( tiers 5 -- 11 ) 5 None 6 Curzon Ashton F.C. United of Manchester Salford City Stockport County 7 Altrincham Ashton United Stalybridge Celtic 8
Atherton Collieries Droylsden Hyde United Mossley Radcliffe Borough Ramsbottom United Trafford 9 Abbey Hey Ashton Athletic Irlam Maine Road West Didsbury &
Chorlton 10 Atherton Laburnum Rovers Chadderton Cheadle Town Daisy Hill Prestwich Heys Stockport Town 11 Ashton Town Eagley Wythenshawe Amateurs Competitions
Manchester Premier Cup Manchester Senior Cup Venues Boundary Park Bower Fold Broadhurst Park Butcher 's Arms Ground City of Manchester Stadium DW Stadium
Edgeley Park Ewen Fields Gigg Lane Macron Stadium Moss Lane Old Trafford Spotland Stadium Other topics Manchester derby Laureus World Sports Award for Team of
the Year winners 2000 : Manchester United F.C.
</item 7>

<item 8>
Manchester United F.C.
Nicknamed `` the Red Devils '' , the club was founded as Newton Heath LYR Football Club in 1878 , changed its name to Manchester United in 1902 and moved to its
current stadium , Old Trafford , in 1910 . Manchester United have won a record 20 League titles , 12 FA Cups , 5 League Cups and a record 21 FA Community
Shields . The club has also won three UEFA Champions Leagues , one UEFA Europa League , one UEFA Cup Winners ' Cup , one UEFA Super Cup , one Intercontinental
Cup and one FIFA Club World Cup . In 1998 -- 99 , the club became the first in the history of English football to achieve the treble of the Premier League , the
FA Cup and the UEFA Champions League . In 2016 -- 17 , by winning the UEFA Europa League , they became one of five clubs to have won all three main UEFA club
competitions . In addition , they became the only professional English club to have won every ongoing honour available to the first team that is organised by a
national or international governing body . The 1958 Munich air disaster claimed the lives of eight players .
</item 8>

<item 9>
Manchester United F.C.
on BBC Sport : Club news -- Recent results and fixtures Manchester United at Sky Sports Manchester United at Premier League Manchester United Football Club
Players Managers Reserves & Academy Records & statistics Current season History 1878 -- 1945 1945 -- 1969 1969 -- 1986 1986 -- 2013 Seasons Europe Munich air
disaster Munich air disaster Busby Babes United Home stadium North Road ( 1878 -- 1893 ) Bank Street ( 1893 -- 1910 ) Old Trafford ( 1910 -- present ) Stretford
End Sir Matt Busby Way Training ground The Cliff Trafford Training Centre Players 100 + appearances 25 -- 99 appearances 1 -- 24 appearances Sir Matt Busby
Player of the Year Rivalries League record by opponent Manchester derby Liverpool rivalry Leeds United rivalry Arsenal rivalry Supporters Independent Manchester
United Supporters Association Manchester United Supporters ' Trust Red Army Media MUTV MU Radio Inside United Red Issue Red News Related articles Fergie 's
Fledglings United Trinity F.C. United of Manchester Glazer ownership Mascots 1915 British football betting scandal Railway station The Class of ' 92 Book :
Manchester United F.C. Category : Manchester United F.C. Portal
</item 9>

<item 10>
Manchester United F.C.
The record was pushed steadily upwards before reaching its peak on 31 March 2007 , when 76,098 spectators saw Manchester United beat Blackburn Rovers 4 -- 1 ,
with just 114 seats ( 0.15 per cent of the total capacity of 76,212 ) unoccupied . In 2009 , reorganisation of the seating resulted in a reduction of capacity
by 255 to 75,957 . Manchester United has the second highest average attendance of European football clubs only behind Borussia Dortmund . Support Manchester
United is one of the most popular football clubs in the world , with one of the highest average home attendance in Europe . The club states that its worldwide
fan base includes more than 200 officially recognised branches of the Manchester United Supporters Club ( MUSC ) , in at least 24 countries . The club takes
advantage of this support through its worldwide summer tours . Accountancy firm and sports industry consultants Deloitte estimate that Manchester United has 75
million fans worldwide , while other estimates put this figure closer to 333 million .
</item 10>
=== END ITEM id=8b347d481d15ef0b ===
