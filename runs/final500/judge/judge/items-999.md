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

=== ITEM id=6f5b550f57cfc257 ===
Gold sentences and facts:
Sentence 0: In the 12th century , luxury increased and bedsteads were made of wood much decorated with inlaid , carved , and painted ornamentation . They also used folding beds , which served as couches by day and had cushions covered with silk laid upon leather . At night a linen sheet was spread and pillows placed , while silk - covered skins served as coverlets . The Carolingian manuscripts show metal bedsteads much higher at the head than at the feet , and this shape continued in use until the 13th century in France , many cushions being added to raise the body to a sloping position . In 12th - century manuscripts , the bedsteads appear much richer , with inlays , carving , and painting , and with embroidered coverlets and mattresses in harmony . Curtains were hung above the bed and a small hanging lamp is often shown .
  Fact 0.0: In the 12th century, luxury increased.
  Fact 0.1: In the 12th century, bedsteads were made of wood.
  Fact 0.2: The wooden bedsteads were much decorated with inlaid, carved, and painted ornamentation.
  Fact 0.3: Folding beds were used.
  Fact 0.4: The folding beds served as couches by day.
  Fact 0.5: The folding beds had cushions covered with silk laid upon leather.
  Fact 0.6: At night a linen sheet was spread on the bed.
  Fact 0.7: At night pillows were placed on the bed.
  Fact 0.8: At night silk-covered skins served as coverlets.
  Fact 0.9: Carolingian manuscripts show metal bedsteads much higher at the head than at the feet.
  Fact 0.10: The shape of the metal bedsteads with a higher head continued in use until the 13th century in France.
  Fact 0.11: Many cushions were added to raise the body to a sloping position.
  Fact 0.12: In 12th-century manuscripts, bedsteads appear much richer.
  Fact 0.13: The richer 12th-century bedsteads have inlays, carving, and painting.
  Fact 0.14: The richer 12th-century bedsteads have embroidered coverlets and mattresses in harmony.
  Fact 0.15: Curtains were hung above the bed.
  Fact 0.16: A small hanging lamp is often shown.

Retrieved items:
<item 1>
Bed
Silk , velvet , and even cloth of gold were frequently used . Inventories from the beginning of the 14th century give details of these hangings lined with fur
and richly embroidered . It was then that the Four poster bed ( also known as a tester bed ) made its first appearance , the bed being slung from the ceiling or
fastened to the walls , a form which developed later into a room within a room , shut in by double curtains , sometimes even to exclude all drafts . The space
between bed and wall was called the ruelle , and very intimate friends were received there . The 14th century is also the time when feather beds became highly
prized possessions . In the 15th century beds became very large , reaching 7 to 8 feet ( 2.1 to 2.4 m ) by 6 to 7 feet ( 1.8 to 2.1 m ) . The mattresses were
often filled with pea - shucks , straw , or feathers .
</item 1>

<item 2>
Bed
Ancient Romans had various kinds of beds for repose . These included : lectus cubicularis , or chamber bed , for normal sleeping lectus genialis , the marriage
bed , it was much decorated , and was placed in the atrium opposite the door lectus discubitorius , or table bed , on which they ate -- for they ate while lying
on their left sides -- there usually being three people to one bed , with the middle place accounted the most honorable position lectus lucubratorius , for
studying and a lectus funebris , or emortualis , on which the dead were carried to the pyre Medieval Europe The ancient Germans lay on the floor on beds of
leaves covered with skins , or in a kind of shallow chest filled with leaves and moss . In the early Middle Ages they laid carpets on the floor or on a bench
against the wall , placed upon them mattresses stuffed with feathers , wool , or hair , and used skins as a covering . Curtains were hung from the ceiling or
from an iron arm projecting from the wall . They appear to have generally lain naked in bed , wrapping themselves in large linen sheets which were stretched
over the cushions .
</item 2>

<item 3>
Bed
So much gold was used that the velvet scarcely showed . Napoleon I 's bed In the 18th century feather pillows were first used as coverings in Germany , which in
the fashions of the bed and the curious etiquette connected with the bedchamber followed France for the most part . The beds were a la duchesse , but in France
itself there was great variety both of name and shape . The custom of the `` bed of justice '' upon which the king of France reclined when he was present in
parliament , the princes being seated , the great officials standing , and the lesser officials kneeling , was held to denote the royal power even more than the
throne . Louis XI is credited with its first use and the custom lasted till the end of the monarchy . In the chambre de parade , where the ceremonial bed was
placed , certain persons , such as ambassadors or great lords , whom it was desired to honour , were received in a more intimate fashion than the crowd of
courtiers . At Versailles women received their friends in their beds , both before and after childbirth , during periods of mourning , and even directly after
marriage -- in fact in any circumstances which were thought deserving of congratulation or condolence .
</item 3>

<item 4>
Bed
The mattresses were often filled with pea - shucks , straw , or feathers . At this time great personages were in the habit of carrying most of their property
about with them , including beds and bed - hangings , and for this reason the bedsteads were for the most part mere frameworks to be covered up ; but about the
beginning of the 16th century bedsteads were made lighter and more decorative , since the lords remained in the same place for longer periods . Renaissance and
modern Europe In the 17th century , which has been called `` the century of magnificent beds '' , the style a la duchesse , with tester and curtains only at the
head , replaced the more enclosed beds in France , though they lasted much longer in England . Louis XIV had an enormous number of sumptuous beds , as many as
413 being described in the inventories of his palaces . Some of them had embroideries enriched with pearls , and figures on a silver or golden ground . The
great bed at Versailles had crimson velvet curtains on which `` The Triumph of Venus '' was embroidered . So much gold was used that the velvet scarcely showed
.
</item 4>

<item 5>
Bed
Bed - wikipedia Bed This is the latest accepted revision , reviewed on 10 January 2018 . Jump to : navigation , search For other uses , see Bed ( disambiguation
) . Bedroom on the Detmold Open - air Museum premises A bed is a piece of furniture which is used as a place to sleep or relax . Most modern beds consist of a
soft , cushioned mattress on a bed frame , the mattress resting either on a solid base , often wood slats , or a sprung base . Many beds include a box spring
inner - sprung base , which is a large mattress - sized box containing wood and springs that provide additional support and suspension for the mattress . Beds
are available in many sizes , ranging from infant - sized bassinets and cribs , to small beds for a single person or adult , to large queen and king - size beds
designed for two people . While most beds are single mattresses on a fixed frame , there are other varieties , such as the murphy bed , which folds into a wall
, the sofa bed , which folds out of a sofa , and the bunk bed , which provides two mattresses on two tiers .
</item 5>

<item 6>
Bed
Most often , the difference is 2 - 5 inches . Notable examples The Great Bed of Ware , one of the largest beds in the world One of the largest beds in the world
is the Great Bed of Ware , made in about 1580 . It is 3.26 metres ( 10.7 ft ) wide , 3.38 metres ( 11.1 ft ) long . The bed is mentioned by Shakespeare in
Twelfth Night . It is now in the Victoria and Albert Museum ( V&A ) in London . Another bed in the V&A is the Golden Bed created by William Burges in 1879 . In
1882 , an Indian Maharajah had a bed made of solid silver . At each corner of the bed there was a life - sized statue of a naked woman holding a fan . When the
Maharajah lay on the bed , his weight started a mechanism that made the women wave their fans . In 1865 , a convertible bed in the form of an upright piano was
available , which could provide home entertainment while saving space . Types This section does not cite any sources . Please help improve this section by
adding citations to reliable sources .
</item 6>

<item 7>
Bed
Southampton Medieval Merchant 's House bedroom In the 12th century , luxury increased and bedsteads were made of wood much decorated with inlaid , carved , and
painted ornamentation . They also used folding beds , which served as couches by day and had cushions covered with silk laid upon leather . At night a linen
sheet was spread and pillows placed , while silk - covered skins served as coverlets . The Carolingian manuscripts show metal bedsteads much higher at the head
than at the feet , and this shape continued in use until the 13th century in France , many cushions being added to raise the body to a sloping position . In
12th - century manuscripts , the bedsteads appear much richer , with inlays , carving , and painting , and with embroidered coverlets and mattresses in harmony
. Curtains were hung above the bed and a small hanging lamp is often shown . In the 14th century the woodwork became of less importance , generally being
entirely covered by hangings of rich materials . Silk , velvet , and even cloth of gold were frequently used .
</item 7>

<item 8>
Bed
Contents ( hide ) 1 History 1.1 Ancient world 1.2 Medieval Europe 1.3 Renaissance and modern Europe 2 Bed sizes 3 Notable examples 4 Types 5 Frames 6 See also 7
References 8 External links History Ancient world Tutankhamun 's gilded bed from the 14th century BC Early beds were little more than piles of straw or some
other natural material ( e.g. a heap of palm leaves , animal skins , or dried bracken ) . An important change was raising them off the ground , to avoid drafts
, dirt , and pests . Bedding dated around to 3600 BC was discovered in Sibudu Cave , South Africa . The bedding consists of sedge and other monocotyledons
topped with the leaves of Cryptocarya woodii Engl . Beds found in a preserved northern Scottish village , which were raised boxes made of stone and likely
topped with comfortable fillers , were dated to between 3200 BC and 2200 BC . The Egyptians had high bedsteads which were ascended by steps , with bolsters or
pillows , and curtains to hang around .
</item 8>

<item 9>
Bed
During the 17th century this curious custom became general , perhaps to avoid the tiresome details of etiquette . Portable beds were used in high society in
France till the end of the Ancien Régime . The earliest of which mention has been found belonged to Charles the Bold . They had curtains over a light framework
, and were in their way as fine as the stationary beds . Iron beds appear in the 18th century ; the advertisements declare them as free from the insects which
sometimes infested wooden bedsteads . Elsewhere , there was also the closed bed with sliding or folding shutters , and in England -- where beds were commonly
quite simple in form -- the four poster was the usual citizen 's bed until the middle of the 19th century . Bed sizes Main article : Bed size Bed sizes vary
considerably around the world , with most countries having their own standards and terminology . While the `` double '' size appears to be standard among
English speaking countries , based on the imperial measurement of 4 ft 6 in by 6 ft 3 in ( 137 cm x 190 cm ) , the sizes for other bed types tend to vary .
</item 9>

<item 10>
Bed
Please help improve this section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( April 2015 ) ( Learn how and
when to remove this template message ) See also : Category : Beds Lit à la Polonaise ( Polish style bed ) , Royal Castle in Warsaw , 18th century . Patent #
322,177 , on 14 July 1885 issued to Sarah E. Goode for a cabinet bed Drawing of a candle - lit mourning bed ( Trauergerüst ) for abbess Franziska Christine von
Pfalz - Sulzbach , 1776 Chinese style beds There are many varieties of beds : An adjustable bed is a bed that can be adjusted to a number of different positions
. Most hospital beds are adjustable , so that a patient can have different parts of her body elevated for medical reasons . Some people have adjustable beds in
private homes . Some adjustable beds are designed for couples ; they use two separate mattresses and adjustment mechanisms . This permits one partner to be
lying flat to sleep while the other has the head and shoulders elevated to watch television or read .
</item 10>
=== END ITEM id=6f5b550f57cfc257 ===

=== ITEM id=77a86cbdc7b9f6a4 ===
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
External links Wikimedia Commons has media related to Philadelphia Eagles . Official website Philadelphia Eagles Founded in 1933 Based and headquartered in
Philadelphia , Pennsylvania Franchise Franchise History Seasons Coaches Quarterbacks Players Stadiums Baker Bowl Philadelphia Municipal Stadium Connie Mack
Stadium Franklin Field Veterans Stadium Lincoln Financial Field Culture Fly , Eagles Fly Swoop Curse of Billy Penn The Lombardi Curse Invincible Philadelphia
Sports Hall of Fame Matt Guokas Sr . Dan Baker Cheerleaders Silver Linings Playbook The Garbage Picking Field Goal Kicking Philadelphia Phenomenon Lore
Frankford Yellow Jackets Pennsylvania Keystoners `` Pennsylvania Polka '' Steagles `` Happy Hundred '' Miracle at the Meadowlands Buddy Ball Fog Bowl Bounty
Bowl series Body Bag Game 4th and 26 Miracle at the New Meadowlands Philly Special Rivalries Dallas Cowboys New York Giants Washington Redskins Pittsburgh
Steelers Division championships ( 13 ) 1947 1948 1949 1980 1988 2001 2002 2003 2006 2013 2017 Conference championships ( 4 ) 1960 1980 2017 League championships
( 4 ) 1948 1949 1960 2017 ( LII ) Retired numbers 5
</item 1>

<item 2>
Philadelphia Eagles
Foles had his best game since week 15 and threw for 352 passing yards and 3 touchdowns including an impressive flea flicker touchdown pass . This win marked the
franchise 's third Super Bowl appearance and a berth in Super Bowl LII for a 2005 rematch against Tom Brady and the New England Patriots . With Foles at the
helm , the Eagles started off the game strong , leading the Patriots 22 -- 12 at halftime . In the second half the Patriots gained momentum and for the first
time took the lead in the 4th quarter 33 -- 32 . The Eagles rallied back and scored an 11 - yard Touchdown to tight end Zach Ertz . Later the Eagles converted
on a 46 - yard field goal by Jake Elliott to lead 41 -- 33 . On the final play of the game with 9 seconds left , Brady threw a hail - mary pass directed toward
tight end Rob Gronkowski which was bobbled by Patrick Robinson before being dropped in the end - zone as time expired , securing the win for the Eagles 41 -- 33
, and ending the 58 - year long championship drought .
</item 2>

<item 3>
Philadelphia Eagles
2002 , 2003 , 2004 , 2006 , 2008 , 2009 , 2010 , 2013 , 2017 Home fields Baker Bowl ( 1933 -- 1935 ) Philadelphia Municipal Stadium ( 1936 -- 1939 , 1941 )
Connie Mack Stadium ( 1940 , 1942 -- 1957 ) Franklin Field ( 1958 -- 1970 ) Veterans Stadium ( 1971 -- 2002 ) Lincoln Financial Field ( 2003 -- present ) The
Philadelphia Eagles are a professional American football franchise based in Philadelphia , Pennsylvania . The Eagles compete in the National Football League (
NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . They are Super Bowl champions , having won Super Bowl LII , their
fourth NFL title , after winning in 1948 , 1949 , and 1960 . The franchise was established in 1933 as a replacement for the bankrupt Frankford Yellow Jackets ,
when a group led by Bert Bell secured the rights to an NFL franchise in Philadelphia .
</item 3>

<item 4>
Philadelphia Eagles
This would be Philadelphia 's first Super Bowl win and their fourth league championship ( their first championship since 1960 ) . Foles won Super Bowl MVP going
28 -- 43 with 373 passing yards , 3 touchdowns , and an interception . Nick Foles became the first backup quarterback to start and win a Super Bowl since his
opponent Tom Brady won as the backup for Drew Bledsoe in 2001 .
</item 4>

<item 5>
Philadelphia Eagles
In 2003 , they won the NFC first seed , but Westbrook went down in Week 17 , culminating in a loss to the Carolina Panthers in their 3rd straight NFC
Championship . 2004 season and Super Bowl XXXIX In 2004 , the Philadelphia Eagles had their best season since 1960 , going 13 -- 1 before resting their starters
and losing their next 2 , clinching the 1st seed for the second year in a row . McNabb set career highs , completing 64 % of his passes for 3,875 yards , though
he did n't play all 16 games . McNabb became the first quarterback ever to throw more than 30 touchdowns and fewer than 10 interceptions in a season . His
success could be attributed to the fact that he had a reliable receiver , Terrell Owens , who got 1,200 yards and 14 touchdowns in 14 games . After beating the
Vikings and Falcons the Eagles advanced to Super Bowl XXXIX , where they dueled the New England Patriots .
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
Season records Main article : List of Philadelphia Eagles seasons Regular season record ( all - time ) : 568 -- 594 -- 26 Playoff record ( all - time ) : 22 --
21 ( last appearance after beating New England Patriots in Super Bowl LII ) Most points in a season : 474 points ( 2014 ) NFL championships won : 4 ( 3 before
the 1967 NFL - AFL merger created the Super Bowl ) Super Bowls won : 1 out of 3 appearances Passing leader ( all - time ) : Donovan McNabb -- 32,873 yards
Rushing leader ( all - time ) : LeSean McCoy -- 6,792 yards Receiving leader ( all - time ) : Harold Carmichael -- 8,978 yards Winningest coach ( all - time ) :
Andy Reid -- 130 wins Top player by approximate value ( all - time ) : Donovan McNabb -- 126 AV Rivalries New York Giants Main article : Eagles -- Giants
rivalry One of the NFL 's oldest , this rivalry began on October 15 , 1933 when the Giants defeated the newly founded Eagles 56 -- 0 .
</item 7>

<item 8>
Graph query row:
organization: Philadelphia Eagles
event: Super Bowl LII
date: 2018
result: None
outcome: None
</item 8>

<item 9>
Super Bowl XXXIX
The Eagles were trying to win their first NFL title since 1960 and the first championship for the city of Philadelphia since Moses Malone 's `` fo ' , fi ' , fo
' '' during the 76ers run to the 1983 NBA Championship . With President George W. Bush being inaugurated for a second time in January , the Eagles were also
trying to end a losing streak -- teams in the city of Philadelphia had lost six straight championships during presidential inauguration years , beginning with
the 76ers loss in 1977 . The streak included the Eagles in Super Bowl XV in 1981 . 1989 was not included in that streak , but 1977 , 1981 , 1985 ( Flyers ) ,
1993 ( Phillies ) , 1997 ( Flyers ) , and 2001 ( 76ers ) were . Television and entertainment ( edit ) The game was televised in the United States by Fox , with
play - by - play announcer Joe Buck and color commentators Cris Collinsworth and Troy Aikman . Pam Oliver ( Patriots sideline ) and Chris Myers ( Eagles
sideline ) served as sideline reporters .
</item 9>

<item 10>
Philadelphia Eagles
As of the conclusion of the 2016 season , during the Lurie era , the Eagles are 1 -- 4 in conference championship games and 0 -- 1 in Super Bowls . On August 13
, 2009 , the Eagles signed quarterback Michael Vick . On December 6 , 2009 , Andy Reid became only the 5th coach in NFL history to win 100 or more games with a
single team in a single decade . The other four are Tom Landry , Don Shula , Tony Dungy , and Bill Belichick , all Super Bowl winners . McNabb finally had a
complete receiving corps , between first round draft pick Jeremy Maclin , DeSean Jackson 's 1,000 yard season , and Brent Celek ranking among the top 5 tight
ends in the league . Without Brian Dawkins , defensive end Trent Cole stepped up and became the dominant force on defense with 12 sacks , earning him his second
trip to the Pro Bowl and All - Pro honors .
</item 10>
=== END ITEM id=77a86cbdc7b9f6a4 ===

=== ITEM id=c08d5f381c661509 ===
Gold sentences and facts:
Sentence 0: The Sahara ( Arabic : الصحراء الكبرى ‎ , aṣ - ṣaḥrāʼ al - kubrá , ' the Great Desert ' ) is the largest hot desert and the third largest desert in the world after Antarctica and the Arctic . Its area of 9,200,000 square kilometres ( 3,600,000 sq mi ) is comparable to the area of China or the United States . The name ' Sahara ' is derived from dialectal Arabic word for `` desert '' , ṣaḥra ( صحرا / ˈsʕaħra / ) .
  Fact 0.0: The Sahara is a desert.
  Fact 0.1: The Sahara's Arabic name is الصحراء الكبرى, transliterated aṣ-ṣaḥrāʼ al-kubrá, meaning 'the Great Desert'.
  Fact 0.2: The Sahara is the largest hot desert in the world.
  Fact 0.3: The Sahara is the third largest desert in the world, after Antarctica and the Arctic.
  Fact 0.4: The Sahara's area is 9,200,000 square kilometres (3,600,000 sq mi).
  Fact 0.5: The Sahara's area is comparable to the area of China.
  Fact 0.6: The Sahara's area is comparable to the area of the United States.
  Fact 0.7: The name 'Sahara' is derived from a dialectal Arabic word for 'desert'.
  Fact 0.8: The dialectal Arabic word is ṣaḥra (صحرا).

Retrieved items:
<item 1>
Sahara
Oases support some life forms in extremely arid deserts . An intense Saharan dust storm sent a massive dust plume northwestward over the Atlantic Ocean on 2
March 2003 Butte naturally sculpted by the wind The Sahara covers large parts of Algeria , Chad , Egypt , Libya , Mali , Mauritania , Niger , Western Sahara ,
Sudan and Tunisia . It covers 9 million square kilometres ( 3,500,000 sq mi ) , amounting to 31 % of Africa . If all areas with a mean annual precipitation of
less than 250 mm were included , the Sahara would be 11 million square kilometres ( 4,200,000 sq mi ) . It is one of three distinct physiographic provinces of
the African massive physiographic division . The Sahara is mainly rocky hamada ( stone plateaus ) , Ergs ( sand seas - large areas covered with sand dunes )
form only a minor part , but many of the sand dunes are over 180 metres ( 590 ft ) high .
</item 1>

<item 2>
Sahara
Sahara The Great Desert Desert A satellite image of the Sahara by NASA World Wind Countries Algeria , Chad , Egypt , Libya , Mali , Mauritania , Morocco , Niger
, Sudan , Tunisia Highest point Emi Koussi 3,415 m ( 11,204 ft ) - coordinates 19 ° 47 ′ 36 '' N 18 ° 33 ′ 6 '' E ﻿ / ﻿ 19.79333 ° N 18.55167 ° E ﻿ / 19.79333 ;
18.55167 Lowest point Qattara Depression − 133 m ( − 436 ft ) - coordinates 30 ° 0 ′ 0 '' N 27 ° 5 ′ 0 '' E ﻿ / ﻿ 30.00000 ° N 27.08333 ° E ﻿ / 30.00000 ;
27.08333 Length 4,800 km ( 2,983 mi ) , E / W Width 1,800 km ( 1,118 mi ) , N / S Area 9,200,000 km ( 3,552,140 sq mi ) Biome Desert This article contains
Arabic text .
</item 2>

<item 3>
Sahara
The desert comprises much of North Africa , excluding the fertile region on the Mediterranean Sea coast , the Atlas Mountains of the Maghreb , and the Nile
Valley in Egypt and Sudan . It stretches from the Red Sea in the east and the Mediterranean in the north to the Atlantic Ocean in the west , where the landscape
gradually changes from desert to coastal plains . To the south , it is bounded by the Sahel , a belt of semi-arid tropical savanna around the Niger River valley
and the Sudan Region of Sub-Saharan Africa . The Sahara can be divided into several regions including : the western Sahara , the central Ahaggar Mountains , the
Tibesti Mountains , the Aïr Mountains , the Ténéré desert , and the Libyan Desert . For several hundred thousand years , the Sahara has alternated between
desert and savanna grassland in a 41,000 year cycle caused by changes ( `` precession '' ) in the Earth 's axis as it rotates around the sun , which change the
location of the North African Monsoon . It is next expected to become green in about 15,000 years ( 17,000 AD ) .
</item 3>

<item 4>
Sahara
See also Arid Lands Information Network List of deserts List of deserts by area Neolithic Subpluvial Sahara Conservation Fund Sahara Sea Saharan explorers
References Jump up ^ `` Largest Desert in the World '' . Retrieved 30 December 2011 . Jump up ^ Cook , Kerry H. ; Vizy , Edward K. ( 2015 ) . `` Detection and
Analysis of an Amplified Warming of the Sahara Desert '' . Journal of Climate. 28 : 6560 . doi : 10.1175 / JCLI - D - 14 - 00230.1 . Jump up ^ `` Sahara . ''
Online Etymology Dictionary . Douglas Harper , Historian . Retrieved 25 June 2007 . Jump up ^ `` English - Arabic online dictionary '' . Online.ectaco.co.uk. 28
December 2006 . Retrieved 12 June 2010 . Jump up ^ Wehr , Hans ( 1994 ) . A Dictionary of Modern Written Arabic ( Arabic - English ) ( 4th ed . ) . Wiesbaden :
Otto Harrassowitz . p. 589 .
</item 4>

<item 5>
Sahara
The southern limit of the Sahara is indicated botanically by the southern limit of Cornulaca monacantha ( a drought - tolerant member of the Chenopodiaceae ) ,
or northern limit of Cenchrus biflorus , a grass typical of the Sahel . According to climatic criteria , the southern limit of the Sahara corresponds to the 150
mm ( 5.9 in ) isohyet of annual precipitation ( this is a long - term average , since precipitation varies annually ) . Important cities located in the Sahara
include Nouakchott , the capital of Mauritania ; Tamanrasset , Ouargla , Béchar , Hassi Messaoud , Ghardaïa , and El Oued in Algeria ; Timbuktu in Mali ; Agadez
in Niger ; Ghat in Libya ; and Faya - Largeau in Chad . Climate The Sahara is the world 's largest low - latitude hot desert . The area is located in the horse
latitudes under the subtropical ridge , a significant belt of semi-permanent subtropical warm - core high pressure where the air from upper levels of the
troposphere tends to sink towards the ground .
</item 5>

<item 6>
Sahara
Stone arrowheads , knives and scrapers from the era are commonly found . Burial items included pottery , jewelry , farming and hunting equipment , and assorted
foods including dried meat and fruit . Burial in desert environments appears to enhance Egyptian preservation rites , and the dead were buried facing due west .
By 3400 BCE , the Sahara was as dry as it is today , due to reduced precipitation and higher temperatures resulting from a shift in the Earth 's orbit . As a
result of this aridification , it became a largely impenetrable barrier to humans , with the remaining settlements mainly being concentrated around the numerous
oases that dot the landscape . Little trade or commerce is known to have passed through the interior in subsequent periods , the only major exception being the
Nile Valley . The Nile , however , was impassable at several cataracts , making trade and contact by boat difficult . Phoenicians Further information : History
of Western Sahara The people of Phoenicia , who flourished from 1200 -- 800 BCE , created a confederation of kingdoms across the entire Sahara to Egypt .
</item 6>

<item 7>
Sahara
Without proper rendering support , you may see question marks , boxes , or other symbols . Play media This video over the Sahara and the Middle East was taken
by the crew of Expedition 29 on board the International Space Station . Tadrart Acacus desert in western Libya , part of the Sahara . The top image shows the
Safsaf Oasis on the surface of the Sahara . The bottom ( using radar ) is the rock layer underneath , revealing black channels cut by the meandering of an
ancient river that once fed the oasis . The Sahara ( Arabic : الصحراء الكبرى ‎ , aṣ - ṣaḥrāʼ al - kubrá , ' the Great Desert ' ) is the largest hot desert and
the third largest desert in the world after Antarctica and the Arctic . Its area of 9,200,000 square kilometres ( 3,600,000 sq mi ) is comparable to the area of
China or the United States . The name ' Sahara ' is derived from dialectal Arabic word for `` desert '' , ṣaḥra ( صحرا / ˈsʕaħra / ) .
</item 7>

<item 8>
Sahara
The Saharan halophytics cover 54,000 square kilometres ( 21,000 sq mi ) including : the Qattara and Siwa depressions in northern Egypt , the Tunisian salt lakes
of central Tunisia , Chott Melghir in Algeria , and smaller areas of Algeria , Mauritania , and the southern part of Morocco . The Tanezrouft is one of the
harshest regions on Earth as well as one of the hottest and driest parts of the Sahara , with no vegetation and very little life . It is along the borders of
Algeria , Niger ) ) and Mali , west of the Hoggar mountains . Flora and fauna The flora of the Sahara is highly diversified based on the bio-geographical
characteristics of this vast desert . Floristically , the Sahara has three zones based on the amount of rainfall received -- the Northern ( Mediterranean ) ,
Central and Southern Zones . There are two transitional zones -- the Mediterranean - Sahara transition and the Sahel transition zone . The Saharan flora
comprises around 2800 species of vascular plants . Approximately a quarter of these are endemic . About half of these species are common to the flora of the
Arabian deserts .
</item 8>

<item 9>
Sahara
Sahara Overland . Trailblazer Guides . Wade , Lizzie ( 2015 ) . Drones and Satellites Spot Lost Civilizations in Unlikely Places . Science ( American
Association for the Advancement of Science ) . DOI : 10.1126 / science. aaa7864 . External links Find more aboutSaharaat Wikipedia 's sister projects
Definitions from Wiktionary Media from Wikimedia Commons Travel guide from Wikivoyage About Sahara subsurface hydrology and planned usage of the aquifers World
deserts Desert Desertification List of deserts List of deserts by area Africa Algerian Bayuda Blue Chalbi Danakil Djurab Eastern Ferlo Farafra ( White )
Kalahari Libyan Moçâmedes Namib Nubian Nyiri Owami Richtersveld Sahara Tanezrouft Ténéré Western Asia Ad - Dahna Akshi Arabian Aral Karakum Aralkum Badain Jaran
Betpak - Dala Cholistan Dasht - e Kavir Dasht - e Khash Dasht - e Leili Dasht - e Loot Dasht - e Margo Dasht - e Naomid
</item 9>

<item 10>
Sahara
Wind or rare rainfall shape the desert features : sand dunes , dune fields , sand seas , stone plateaus , gravel plains ( reg ) , dry valleys ( wadi ) , dry
lakes ( oued ) , and salt flats ( shatt or chott ) . Unusual landforms include the Richat Structure in Mauritania . Several deeply dissected mountains , many
volcanic , rise from the desert , including the Aïr Mountains , Ahaggar Mountains , Saharan Atlas , Tibesti Mountains , Adrar des Iforas , and the Red Sea hills
. The highest peak in the Sahara is Emi Koussi , a shield volcano in the Tibesti range of northern Chad . The central Sahara is hyperarid , with sparse
vegetation . The northern and southern reaches of the desert , along with the highlands , have areas of sparse grassland and desert shrub , with trees and
taller shrubs in wadis , where moisture collects . In the central , hyperarid region , there are many subdivisions of the great desert : Tanezrouft , the Ténéré
, the Libyan Desert , the Eastern Desert , the Nubian Desert and others .
</item 10>
=== END ITEM id=c08d5f381c661509 ===

=== ITEM id=03423464d8b0b46f ===
Gold sentences and facts:
Sentence 0: `` Knightfall '' is a 1993 - 1994 Batman story arc published by DC Comics . It consists of a trilogy of storylines that ran from 1993 to 1994 , consisting of `` Knightfall '' , `` Knightquest '' , and `` KnightsEnd '' . The story takes place over approximately six months . Bruce Wayne ( Batman ) suffers burnout and is systematically assaulted and crippled by a `` super steroid '' - enhanced genius named Bane . Wayne is replaced as Batman by an apprentice named Jean - Paul Valley ( a.k.a. Azrael ) , who becomes increasingly violent and unstable , tarnishing Batman 's reputation . Eventually , Wayne is healed through paranormal means , and reclaims his role as Batman .
  Fact 0.0: 'Knightfall' is a 1993-1994 Batman story arc published by DC Comics.
  Fact 0.1: Knightfall consists of a trilogy of storylines that ran from 1993 to 1994.
  Fact 0.2: The trilogy consists of 'Knightfall', 'Knightquest', and 'KnightsEnd'.
  Fact 0.3: The story takes place over approximately six months.
  Fact 0.4: Bruce Wayne (Batman) suffers burnout.
  Fact 0.5: Bruce Wayne (Batman) is systematically assaulted and crippled by a 'super steroid'-enhanced genius named Bane.
  Fact 0.6: Wayne is replaced as Batman by an apprentice named Jean-Paul Valley (a.k.a. Azrael).
  Fact 0.7: Jean-Paul Valley becomes increasingly violent and unstable.
  Fact 0.8: Jean-Paul Valley tarnishes Batman's reputation.
  Fact 0.9: Eventually, Wayne is healed through paranormal means.
  Fact 0.10: Eventually, Wayne reclaims his role as Batman.

Retrieved items:
<item 1>
Batman: Knightfall
Art by Mike Gustovich . Bruce reaffirms his partnership with Tim , resolving the tension caused by Bruce 's unwillingness to accept help during the Arkham
prison break . Bruce passes the mantle of Batman to Grayson so he can re-evaluate what it will take to restore his aura of invincibility . This begins the
Prodigal storyline , a reference to Dick Grayson essentially being Bruce 's prodigal son ; Bruce had adopted Dick after his parents were murdered . Because of
the events of the entire arc , considerable time passes before Commissioner Gordon restores his trust in the idea of a Batman working for good . Gordon can tell
that he is not looking at the original Batman ( based on Jean - Paul 's costume and Dick 's height , and the fact that Jean - Paul was more than ready to kill
people ) , and he refuses to place blind trust in a costume after spending so long learning to trust the man .
</item 1>

<item 2>
Batman: Knightfall
Zsasz , Cornelius Stirk , Abattoir , and Amygdala . The scenario creates a rift in the relationship between Robin and Batman , as Batman irrationally seeks to
face the outbreak alone -- in later issues , Robin asks Batman if he is even needed as his sidekick anymore . A later flashback to this time period ( Showcase '
93 # 7 - 8 ) shows Batman pursuing Two - Face alone , being trapped and kidnapped to stand a mock trial ; he is saved only by a rescue attempt from Robin . Bane
breaks Batman 's back in a splash page from Batman # 497 ( July 1993 ) . Art by Jim Aparo . Over the next few issues , Batman becomes weaker and weaker as each
criminal is put away . The rescue of Mayor Krol from the teaming of the Joker and Scarecrow pushes Batman to his mental and physical limits : a dose of
Scarecrow 's fear gas makes him relive the murder of Jason Todd , which he considers to be his greatest failure .
</item 2>

<item 3>
Batman: Knightfall
After this encounter , Bane 's men assault Batman before he himself makes his move and attacks him at Wayne Manor , his home as his alter - ego -- by this time
; Bane had deduced the secret identity of Batman . The fight between Bruce Wayne and Bane is detailed in Batman # 497 . By this time , Batman is so exhausted
from the exertion of catching the escaped criminals that he can not even defend himself ; Bane beats him mercilessly before breaking Wayne 's back over his knee
inside the Batcave below the manor , symbolically `` breaking '' Batman . Bane takes the grievously - wounded Wayne ( still costumed in the Batman outfit )
downtown to Gotham Square and throws him from a rooftop to demonstrate his superiority to the populace . Quick action by Robin and Alfred spares Batman 's life
, but at great cost - he is left a paraplegic . With Batman incapacitated , Bane assumes control of Gotham City 's underworld and takes over several illegal
operations within it . After his defeat , Bruce Wayne enlists the aid of Dr.
</item 3>

<item 4>
Batman: Knightfall
The two characters were quickly added to the cast in the monthly Batman titles , with Azrael being a superhero - in - training who fights alongside Batman ,
while Bane was introduced as a supervillain . Within the regular series , the buildup to `` Knightfall '' begins with a six - issue run in Batman # 484 - 489 (
September 1992 - February 1993 ) , in which Batman ( at the onset of a personal psychological mid-life crisis ) is forced to deal , in rapid succession , with
the returning villain Black Mask and his gang ( who target Bruce Wayne and Lucius Fox ) , a crazed killer called Metalhead , and a sharpshooter assassin hired
by an imprisoned mobster to murder Commissioner Gordon . Batman begins to feel he has lost his edge , especially after his failure to capture Black Mask . He
finds himself unable to meditate or even focus . As Bruce Wayne , he contacts holistic therapist Shondra Kinsolving for treatment . He also assigns Robin ( Tim
Drake ) to train Jean - Paul Valley in detective work to aid them as an ally , hoping to guide Valley 's brainwashing away from making him a villainous threat .
</item 4>

<item 5>
Batman: Knightfall
During the breakout at Arkham Asylum , Batman and Robin both wear a black arm band with the S - shield engraved on it . The two stories involved DC Comics 's
most prominent characters . Similar stories followed for Green Lantern , Wonder Woman , Aquaman , and Green Arrow , with Green Lantern Hal Jordan being driven
insane and replaced with Kyle Rayner , a new Wonder Woman being appointed by Hippolyta , Aquaman losing his hand , and Green Arrow dying in a plane crash as his
long - lost son took over the role . In the opening of the novelisation , KnightFall , Dennis O'Neil stated that part of the reason `` Knightfall '' was written
was due to the recent popularity of more `` ruthless '' heroes such as the Terminator and James Bond in films , as editors were starting to wonder if readers
would prefer a Batman who was willing to kill his opponents . During DC vs. Marvel , Bane attempts to break Captain America 's back in a fight in a similar
maneuver , but is caught off - guard when Captain America 's shield returns to its owner to strike Bane in the back of the head and knock him out .
</item 5>

<item 6>
Batman: Knightfall
It also includes a selection of challenge maps featuring characters and enemies from the storyline . Bane makes reference to `` breaking the bat '' on a few
occasions and in multiplayer , Bane can perform the backbreaker to instantly kill Batman or Robin . In Batman : Arkham Knight , Azrael appears in a sidequest in
which he expresses his interest in becoming Batman 's successor ( though this is the Michael Lane version ) , referencing Azrael replacing Batman in the
Knightfall storyline . In the game 's ending , Batman activates the Knightfall protocol after Scarecrow reveals to the world he 's Bruce Wayne . He is
supposedly killed when he arrives home and Wayne Manor explodes ( though a likely scenario is that he faked his death ) and a new Batman emerged protecting
Gotham 's streets afterwards . Whether this is Bruce or a successor is never shown .
</item 6>

<item 7>
Batman: Knightfall
Television ( edit ) The Flash alluded to Knightfall in the episode `` Enter Zoom . '' The Flash 's enemy Zoom overpowers Barry in their battle and breaks his
back . He then goes to Central City News and the police department holding Barry 's broken body while taunting Central City of his victory similar to what Bane
did to Batman . After Zoom escapes , Barry 's accelerated healing allows him to heal from his damaged back , although he takes some time to recover from the
psychological effects of the beating . Animation ( edit ) The character Bane was quickly added to the roster of villains in Batman : The Animated Series . He
appeared in the episode `` Bane '' as an assassin rather than the mastermind as he was in the comics . He was hired by the mobster Rupert Thorne to eliminate
Batman , and in turn by Thorne 's moll to eliminate Thorne afterwards . Bane eventually fought Batman on board a boat ( where Robin had been kidnapped and tied
up ) , but before he could break his back as he did in the comics , Batman thrusts a crumpled batarang into the controls that inject Bane with Venom .
</item 7>

<item 8>
Batman: Knightfall
Returning to Gotham , Bruce meets with Tim . Even though Jean - Paul disobeyed Bruce 's order to refrain from attacking Bane , Bruce is sufficiently impressed
with Jean - Paul 's results . Bruce decides to retire and allow Jean - Paul to continue as Batman . But when Robin tells Bruce of the circumstances surrounding
Abattoir 's death , Bruce sneaks into the Batcave and demands that Jean - Paul step down . Jean - Paul refuses and tells Bruce to leave the cave and never come
back . To rehabilitate his skills due to his lost reflexes after so long out of action , Bruce asks the famed assassin Lady Shiva to retrain him ( using a mask
to conceal his identity and arguing that she will do it for the same reason that she does anything ; it might be interesting ) . Instead , Shiva pits Bruce
against several vengeful expert martial artists , having killed their master while wearing a distinctive tengu mask that carries a motif of a bat that she
subsequently gives to Batman . Shiva 's caveat is that these attacks will continue indefinitely until Bruce Wayne breaks his vow to never kill .
</item 8>

<item 9>
Batman: Knightfall
This action caused a rapid and uncontrollable feed into Bane 's body before Batman pulls out the tube , stopping a fatal overdose of the drug . The Knightfall
saga was loosely adapted in The Batman episode Traction . In this version , Bane lures the Batman into a trap . Batman overpowers Bane in his normal form , but
when Bane unleashed his more bestial side , he severely injures Batman and breaks his back off screen . Believing him to be dead , Bane attempts to take over
Gotham City . Three weeks after crippling Batman , Bane goes on a rampage through Gotham and knocks out Detective Ellen Yin . As he prepares to kill her ,
Batman interferes wearing a new mechanical Batsuit and battles Bane . As Bane attempts to unmask Batman and kill him , Batman uses an electrical wire and knocks
out Bane with it . Film ( edit ) In Justice League : Doom , Bane makes reference to the storyline when he confronts Bruce Wayne above his parents ' empty graves
, stating : `` When we fought before , I broke the bat . Today ... I break the man . ''
</item 9>

<item 10>
Batman: Knightfall
After his defeat , Bruce Wayne enlists the aid of Dr. Shondra Kinsolving to rehabilitate him and asks Jean - Paul Valley to take up the mantle of Batman so that
Gotham has a protector . Tim Drake argues with Bruce to allow Dick Grayson ( the former Robin and then - current Nightwing ) to become Batman , as he is more
experienced and mentally competent . Bruce replies that Grayson has his own responsibilities and would only take up the mantle of the Bat reluctantly ( Dick
later expressed resentment at not being asked to stand in as Batman ) . Bruce 's rationale for this decision is revealed in later issues - secretly , he does
not want Dick to have to face Bane , as he knows Dick 's character will compel him to try . Indeed , Bruce gives Jean - Paul strict orders never to engage Bane
in combat -- and when Jean - Paul does face Bane , only his modified gloves save Jean - Paul from being thrown to his death .
</item 10>
=== END ITEM id=03423464d8b0b46f ===

=== ITEM id=f1d349d40127a6ea ===
Gold sentences and facts:
Sentence 0: Republic of Doyle is a Canadian comedy - drama television series set in St. John 's , Newfoundland and Labrador which aired on CBC Television from January 6 , 2010 to December 10 , 2014 .
  Fact 0.0: Republic of Doyle is a Canadian comedy-drama television series.
  Fact 0.1: Republic of Doyle is set in St. John's, Newfoundland and Labrador.
  Fact 0.2: Republic of Doyle aired on CBC Television.
  Fact 0.3: Republic of Doyle aired from January 6, 2010 to December 10, 2014.

Retrieved items:
<item 1>
*Republic of Doyle* follows father-and-son private investigators Malachy “Mal” Doyle, 56, and Jake Doyle, who run Doyle and Doyle. Jake’s complicated romance
with Leslie Bennett, an intelligent, sharp-shooting Royal Newfoundland Constabulary inspector and occasional source of case information, grows stronger from
season four despite secrets from her past. In the series finale, Inspector Pickard shoots Leslie; while she is hospitalized, Jake learns she is pregnant with
twins. Season one introduces Jake’s
</item 1>

<item 2>
*Republic of Doyle* is a CBC comedy-drama/police procedural set in St. John’s, following private investigators Jake Doyle, his father Mal, and their family and
colleagues. Created by Allan Hawco and others, it ran six seasons, 77 episodes, from 2010–14, earning awards and nominations and international broadcasts; the
complete series was released on DVD in 2016. The material also mentions *Delocated*’s 30 episodes and *Texas Flip N
</item 2>

<item 3>
*Republic of Doyle* is a six-season series featuring Allan Hawco as Jake Doyle, alongside Seán McGinley (Malachy), Lynda Boyd (Rose), Krystin Pellerin (Leslie),
Mark O’Brien (Des), and Marthe Bernard (Tinny). Recurring cast includes Steve O’Connell, Jonathan Goad, Michelle Nolden, Krista Bridges, and others. The article
covers characters, season summaries, episodes, ratings, awards, DVD
</item 3>

<item 4>
Republic of Doyle
Jump up ^ Newsome , Brad ( 6 January 2012 ) . `` Republic of Doyle , Friday , January 6 '' . The Sydney Morning Herald . Retrieved 17 January 2012 . Jump up ^
Daniel MacEachern ( 2013 - 04 - 17 ) . `` ' Republic of Doyle ' headed for U.S. '' The Telegram . Retrieved 2014 - 11 - 21 . External links ( edit ) Television
in Canada portal Official website Republic of Doyle on IMDb Retrieved from `` https://en.wikipedia.org/w/index.php?title=Republic_of_Doyle&oldid=837889899 ''
Categories : 2010 Canadian television series debuts CBC Television shows Culture of St.
</item 4>

<item 5>
*Republic of Doyle* is a Canadian comedy-drama and police procedural created by Allan Hawco, Malcolm MacRury, and Perry Chafe. Set in St. John’s, Newfoundland
and Labrador, it stars Hawco, Seán McGinley, Lynda Boyd, and others. CBC Television aired its six seasons (77 episodes) from January 6, 2010, to December 10,
2014. Each episode runs about 44 minutes.
</item 5>

<item 6>
Republic of Doyle
Republic of Doyle - wikipedia Republic of Doyle Jump to : navigation , search Republic of Doyle Genre Comedy - drama Police procedural Created by Allan Hawco
Malcolm MacRury Perry Chafe Starring Allan Hawco Seán McGinley Lynda Boyd Krystin Pellerin Mark O'Brien Marthe Bernard Bob Cole Opening theme `` Oh Yeah '' by
Great Big Sea Original language ( s ) English No. of seasons 6 No. of episodes 77 ( list of episodes ) Production Executive producer ( s ) Allan Hawco , John
Vatcher , Michael Levine Location ( s ) St. John 's , Newfoundland and Labrador , Canada Cinematography Malcolm Cross Running time 44 minutes Production company
( s ) Take the Shot Productions Fireworks ( seasons 1 - 2 ) Content Film ( season 3 - 6 ) Distributor Trifecta Entertainment & Media ( United States ) Release
Original network CBC Television Picture format 1080i ( HDTV ) Original release January 6 , 2010 -- December 10 , 2014 External links Website Republic of Doyle
is a Canadian comedy - drama television series set in St.
</item 6>

<item 7>
*Republic of Doyle*’s complete series comprises 77 episodes and was listed for release on September 6, 2016. International broadcasts began in 2010, including
in South Africa, the UK, Australia, Brazil, Estonia, Serbia, and Bulgaria; later premieres included Poland (2012) and Germany (2013). In the United States, it
aired in weekend syndication on CW stations and other outlets; in the Netherlands, it aired weekdays on 13th Street until
</item 7>

<item 8>
*Republic of Doyle* is a Canadian comedy-drama television series produced by Entertainment One and set in St. John’s, Newfoundland and Labrador. It features
Irish-Canadian culture and fictional characters of Irish descent, and ended in 2014.
</item 8>

<item 9>
*Republic of Doyle* is a CBC crime drama set and filmed in St. John’s, Newfoundland and Labrador, airing January 2010–December 2014. Allan Hawco stars as
private investigator Jake Doyle, partnering with his retired-police-officer father Malachy (Seán McGinley) and stepmother Rose (Lynda Boyd). The cast also
includes Krystin Pellerin, Mark O’Brien, and Marthe Bernard. Allan Hawco co-created, wrote,
</item 9>

<item 10>
The excerpt lists two articles about *Republic of Doyle*: a 2012 Sydney Morning Herald piece and a 2013 Telegram article reporting the show was headed for the
U.S. It also links to the official website and IMDb, and categorizes the series as a 2010 Canadian television debut and a CBC Television show.
</item 10>
=== END ITEM id=f1d349d40127a6ea ===

=== ITEM id=94c4d1fd55f5690a ===
Gold sentences and facts:
Sentence 0: Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother . Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New '' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti .
  Fact 0.0: Tracy McConnell is better known as 'The Mother'.
  Fact 0.1: Tracy McConnell is the title character from the CBS television sitcom How I Met Your Mother.
  Fact 0.2: The sitcom is narrated by Future Ted.
  Fact 0.3: The sitcom tells the story of how Ted Mosby met The Mother.
  Fact 0.4: Tracy McConnell appears in 8 episodes.
  Fact 0.5: Tracy McConnell's episodes run from 'Lucky Penny' to 'The Time Travelers'.
  Fact 0.6: Tracy McConnell appears as an unseen character in those episodes.
  Fact 0.7: Tracy McConnell was first seen fully in 'Something New'.
  Fact 0.8: Tracy McConnell was promoted to a main character in season 9.
  Fact 0.9: The Mother is played by Cristin Milioti.

Retrieved items:
<item 1>
The Mother (How I Met Your Mother)
While this incident made Cindy realise that she is a lesbian , it also made the Mother decide to go back into dating , as the kiss was her first in a long time
. Some time after this , a man named Darren approaches The Mother and is welcomed into her band named Superfreakonomics . Darren gradually takes over the band .
In the season 6 opener `` Big Days '' it is revealed Ted meets his future wife `` the day of '' the wedding at which he is the best man . In the episode ``
False Positive '' Robin asks Ted to be her future best man , should she ever get married . In the episode `` Challenge Accepted '' , it is revealed that Ted
meets the mother of his children the day of Barney 's wedding . In the last episode of season 7 , `` The Magician 's Code '' it is shown that Barney will marry
Robin , and Ted will meet the mother `` the day of '' their wedding . On the premiere of season 8 , Ted 's wife appears after Barney and Robin 's wedding ,
outside at the `` Farhampton '' station while holding a yellow umbrella and her bass guitar .
</item 1>

<item 2>
The Mother (How I Met Your Mother)
In the video , future Ted is heard saying , `` ... When I think how lucky I am to wake up next to your mom every morning , I ca n't help but be amazed how easy
it all really was ... '' , indirectly stating that The Mother is alive . The video ends right after the train passes at Farhampton station and credits start
rolling , implying that Ted never went back to Robin as he lived a successful married life with Tracy Mosby . Contents ( hide ) 1 Casting 2 Character history
2.1 Name 3 Death 4 Notes 5 References Casting ( edit ) During its first eight seasons , the successful sitcom How I Met Your Mother often hinted at the unseen
character of The Mother . Well - known actresses often made guest appearances on the show . Many fans expected that another would play one of the most - wanted
roles in Hollywood , but creators Carter Bays and Craig Thomas wanted an unknown .
</item 2>

<item 3>
The Mother (How I Met Your Mother)
After they start talking , Mitch encourages her to pursue her dreams . The Mother expresses her desire to end poverty by taking up economics in college . On his
first day of teaching as Professor Mosby , as seen in the season 4 finale `` The Leap '' , he is seen in front of the classroom of students , one of which
Future Ted says is the titular mother . But in the first episode of season 5 , `` Definitions '' , it is revealed that he was actually in the wrong classroom --
Economics instead of Architecture . At the same time in `` How Your Mother Met Me '' , the Mother sits her first session in Economics 305 and meets another
graduate student named Cindy ( Rachel Bilson ) , whom she offers to move in with her as her roommate . They see Ted enter the room , but when he announces the
subject , The Mother thinks she is in the wrong room and runs off . She heads back to the room after seeing Ted scramble to his actual classroom . Later , in ``
Girls Versus Suits '' , Ted dates Cindy , not knowing that her roommate is his future wife .
</item 3>

<item 4>
The Mother (How I Met Your Mother)
In the season 8 episode `` Band or DJ ? , '' Ted runs into Cindy on the subway and tells her that the band Barney and Robin hired to play at their wedding
cancelled at the last minute . The end result of the encounter is that Cindy 's ( now ex - ) roommate 's band plays at Barney and Robin 's wedding . The Mother
is first shown meeting Louis in `` How Your Mother Met Me '' as she is left to carry the band equipment while the now - lead band member Darren talks to his
fans . Later at MacLaren 's Pub , she tells him she 's not yet ready to date . Louis asks her to give him a call if she changes her mind and they begin dating
not long after . The Mother meets all of Ted 's best friends ( Barney , Lily , Marshall and Robin ) before she meets him . The Mother is responsible for
convincing Barney to pursue Robin , as revealed through a flashback in `` Platonish '' . In `` The Locket '' , Tracy meets Lily on a train journey .
</item 4>

<item 5>
The Mother (How I Met Your Mother)
Tracy McConnell appears in 8 episodes from `` Lucky Penny '' to `` The Time Travelers '' as an unseen character ; she was first seen fully in `` Something New
'' and was promoted to a main character in season 9 . The Mother is played by Cristin Milioti . The story of how Ted met The Mother is the framing device behind
the series ; many facts about her are revealed throughout the series , including the fact that Ted once unwittingly owned her umbrella before accidentally
leaving it behind in her apartment . Ted and The Mother meet at the Farhampton train station following Barney Stinson and Robin Scherbatsky 's wedding ; this
scene is shown in `` Last Forever '' , the series finale . The Mother 's death from an unspecified terminal illness in 2024 , also revealed in the series finale
, received a mixed reaction from fans . An alternate ending was released in the ninth season DVD . In the alternate ending , Tracy Mosby is still living when
Ted is telling the story in 2030 .
</item 5>

<item 6>
The Mother (How I Met Your Mother)
In a flashforward in `` The Lighthouse '' , Ted proposes to the Mother at the top of the lighthouse near Farhampton Inn . She immediately accepts . In another
flashfoward in `` Unpause '' , the Mother is revealed to be pregnant with their second child , Luke , in the year 2017 . She goes into labor while she and Ted
are staying at Farhampton . Name ( edit ) The Mother 's real name is not revealed until the series finale , `` Last Forever '' . When Ted meets her at the
Farhampton train station , she reveals that her name is Tracy McConnell . In the season 1 episode `` Belly Full of Turkey '' , Ted meets a stripper named Tracy
and says `` ... that , kids , is the true story of how I met your mother '' . The children react in surprise and appear to believe Ted before he admits he is
joking , which led some fans to correctly guess that The Mother 's name is Tracy .
</item 6>

<item 7>
The Mother (How I Met Your Mother)
Rolling Stone . Retrieved November 16 , 2014 . ( hide ) How I Met Your Mother Episodes Season 1 Season 2 Season 3 Season 4 Season 5 Season 6 Season 7 Season 8
Season 9 Characters Ted Mosby Marshall Eriksen Lily Aldrin Barney Stinson Robin Scherbatsky The Mother ( Tracy McConnell ) Soundtracks How I Met Your Music
Related articles Awards and nominations The Bro Code Cheerleader effect `` Nothing Suits Me Like a Suit '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=The_Mother_(How_I_Met_Your_Mother)&oldid=802354471 '' Categories : How I Met Your Mother characters Fictional
characters introduced in 2013 Fictional female musicians Hidden categories : Pages using deprecated image syntax Pages using infobox character with unknown
parameters Talk Contents About Wikipedia Español Italiano Magyar Tiếng Việt Edit links This page was last edited on 25 September 2017 , at 17 : 03 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 7>

<item 8>
The Mother (How I Met Your Mother)
Jump up ^ `` Pilot '' . How I Met Your Mother . Season 1 . Episode 1 . September 19 , 2005 . CBS . ^ Jump up to : Kuchman , Bill ( April 4 , 2014 ) . `` ' How I
Met Your Mother ' ' Last Forever ' Recap : A Divisive Finale and What It Means for ' HIMYM ' '' . Retrieved July 7 , 2014 . Jump up ^ Rullo , Samantha ( 10
March 2014 ) . `` What is The Mother 's Name on ' How I Met Your Mother ' ? Episode Title Might Hold the Secret '' . Retrieved July 7 , 2014 . Jump up ^ ``
HIMYM Tomatometer Watch : Series Finale Currently Rotten - Majority of critics disappointed by ' Last Forever ' '' . Rotten Tomatoes. April 2 , 2014 . Retrieved
April 3 , 2014 .
</item 8>

<item 9>
The Mother (How I Met Your Mother)
The Mother ( How I Met Your Mother ) - wikipedia The Mother ( How I Met Your Mother ) Jump to : navigation , search Tracy McConnell How I Met Your Mother
character The Mother appearing in `` The Locket '' First appearance `` Lucky Penny ( unseen ) '' `` Something New '' ( seen ) Last appearance `` Last Forever ''
Created by Carter Bays Craig Thomas Portrayed by Cristin Milioti Information Aliases The Mother Gender Female Spouse ( s ) Ted Mosby Significant other ( s ) Max
( deceased former boyfriend ) Louis ( ex-boyfriend ) Children Penny Mosby ( daughter , born in 2015 , played by Lyndsy Fonseca ) Luke Mosby ( son , born in 2017
, played by David Henrie ) Nationality American Tracy McConnell , better known as `` The Mother '' , is the title character from the CBS television sitcom How I
Met Your Mother . The show , narrated by Future Ted , tells the story of how Ted Mosby met The Mother .
</item 9>

<item 10>
The Mother (How I Met Your Mother)
Using Anne Hathaway and Amy Adams as examples , Thomas said that `` We did n't want it to be a big famous star because we did n't want the wider audience to
have associations with whatever actress this would be ... The whole idea is that Ted 's never seen this woman before , so it better feel that way to the
audience '' , similar to how Cobie Smulders being cast as Robin Scherbatsky had `` kept the show alive '' when it began . Bays and Thomas also did not want a
large casting call . They chose Cristin Milioti after seeing her on 30 Rock and Once ; her musical ability was also helpful , as The Mother had been described
as a band member . After competing for the role against at least two others , Milioti filmed her first scene -- for the last episode of season 8 -- having never
watched How I Met Your Mother ; she recalled , `` I had ignorance on my side . So I did n't know what it meant . '' Milioti learned of the character 's
importance only after binge watching the show during the summer .
</item 10>
=== END ITEM id=94c4d1fd55f5690a ===
