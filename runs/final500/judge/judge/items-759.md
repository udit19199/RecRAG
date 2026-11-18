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

=== ITEM id=d087ba366388d787 ===
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
Ancient Romans had various kinds of beds for repose . These included : lectus cubicularis , or chamber bed , for normal sleeping lectus genialis , the marriage
bed , it was much decorated , and was placed in the atrium opposite the door lectus discubitorius , or table bed , on which they ate -- for they ate while lying
on their left sides -- there usually being three people to one bed , with the middle place accounted the most honorable position lectus lucubratorius , for
studying and a lectus funebris , or emortualis , on which the dead were carried to the pyre Medieval Europe The ancient Germans lay on the floor on beds of
leaves covered with skins , or in a kind of shallow chest filled with leaves and moss . In the early Middle Ages they laid carpets on the floor or on a bench
against the wall , placed upon them mattresses stuffed with feathers , wool , or hair , and used skins as a covering . Curtains were hung from the ceiling or
from an iron arm projecting from the wall . They appear to have generally lain naked in bed , wrapping themselves in large linen sheets which were stretched
over the cushions .
</item 1>

<item 2>
Bed
Southampton Medieval Merchant 's House bedroom In the 12th century , luxury increased and bedsteads were made of wood much decorated with inlaid , carved , and
painted ornamentation . They also used folding beds , which served as couches by day and had cushions covered with silk laid upon leather . At night a linen
sheet was spread and pillows placed , while silk - covered skins served as coverlets . The Carolingian manuscripts show metal bedsteads much higher at the head
than at the feet , and this shape continued in use until the 13th century in France , many cushions being added to raise the body to a sloping position . In
12th - century manuscripts , the bedsteads appear much richer , with inlays , carving , and painting , and with embroidered coverlets and mattresses in harmony
. Curtains were hung above the bed and a small hanging lamp is often shown . In the 14th century the woodwork became of less importance , generally being
entirely covered by hangings of rich materials . Silk , velvet , and even cloth of gold were frequently used .
</item 2>

<item 3>
Bed
Please help improve this section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( April 2015 ) ( Learn how and
when to remove this template message ) See also : Category : Beds Lit à la Polonaise ( Polish style bed ) , Royal Castle in Warsaw , 18th century . Patent #
322,177 , on 14 July 1885 issued to Sarah E. Goode for a cabinet bed Drawing of a candle - lit mourning bed ( Trauergerüst ) for abbess Franziska Christine von
Pfalz - Sulzbach , 1776 Chinese style beds There are many varieties of beds : An adjustable bed is a bed that can be adjusted to a number of different positions
. Most hospital beds are adjustable , so that a patient can have different parts of her body elevated for medical reasons . Some people have adjustable beds in
private homes . Some adjustable beds are designed for couples ; they use two separate mattresses and adjustment mechanisms . This permits one partner to be
lying flat to sleep while the other has the head and shoulders elevated to watch television or read .
</item 3>

<item 4>
Bed
During the 17th century this curious custom became general , perhaps to avoid the tiresome details of etiquette . Portable beds were used in high society in
France till the end of the Ancien Régime . The earliest of which mention has been found belonged to Charles the Bold . They had curtains over a light framework
, and were in their way as fine as the stationary beds . Iron beds appear in the 18th century ; the advertisements declare them as free from the insects which
sometimes infested wooden bedsteads . Elsewhere , there was also the closed bed with sliding or folding shutters , and in England -- where beds were commonly
quite simple in form -- the four poster was the usual citizen 's bed until the middle of the 19th century . Bed sizes Main article : Bed size Bed sizes vary
considerably around the world , with most countries having their own standards and terminology . While the `` double '' size appears to be standard among
English speaking countries , based on the imperial measurement of 4 ft 6 in by 6 ft 3 in ( 137 cm x 190 cm ) , the sizes for other bed types tend to vary .
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
The mattresses were often filled with pea - shucks , straw , or feathers . At this time great personages were in the habit of carrying most of their property
about with them , including beds and bed - hangings , and for this reason the bedsteads were for the most part mere frameworks to be covered up ; but about the
beginning of the 16th century bedsteads were made lighter and more decorative , since the lords remained in the same place for longer periods . Renaissance and
modern Europe In the 17th century , which has been called `` the century of magnificent beds '' , the style a la duchesse , with tester and curtains only at the
head , replaced the more enclosed beds in France , though they lasted much longer in England . Louis XIV had an enormous number of sumptuous beds , as many as
413 being described in the inventories of his palaces . Some of them had embroideries enriched with pearls , and figures on a silver or golden ground . The
great bed at Versailles had crimson velvet curtains on which `` The Triumph of Venus '' was embroidered . So much gold was used that the velvet scarcely showed
.
</item 6>

<item 7>
Bed
So much gold was used that the velvet scarcely showed . Napoleon I 's bed In the 18th century feather pillows were first used as coverings in Germany , which in
the fashions of the bed and the curious etiquette connected with the bedchamber followed France for the most part . The beds were a la duchesse , but in France
itself there was great variety both of name and shape . The custom of the `` bed of justice '' upon which the king of France reclined when he was present in
parliament , the princes being seated , the great officials standing , and the lesser officials kneeling , was held to denote the royal power even more than the
throne . Louis XI is credited with its first use and the custom lasted till the end of the monarchy . In the chambre de parade , where the ceremonial bed was
placed , certain persons , such as ambassadors or great lords , whom it was desired to honour , were received in a more intimate fashion than the crowd of
courtiers . At Versailles women received their friends in their beds , both before and after childbirth , during periods of mourning , and even directly after
marriage -- in fact in any circumstances which were thought deserving of congratulation or condolence .
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
Most often , the difference is 2 - 5 inches . Notable examples The Great Bed of Ware , one of the largest beds in the world One of the largest beds in the world
is the Great Bed of Ware , made in about 1580 . It is 3.26 metres ( 10.7 ft ) wide , 3.38 metres ( 11.1 ft ) long . The bed is mentioned by Shakespeare in
Twelfth Night . It is now in the Victoria and Albert Museum ( V&A ) in London . Another bed in the V&A is the Golden Bed created by William Burges in 1879 . In
1882 , an Indian Maharajah had a bed made of solid silver . At each corner of the bed there was a life - sized statue of a naked woman holding a fan . When the
Maharajah lay on the bed , his weight started a mechanism that made the women wave their fans . In 1865 , a convertible bed in the form of an upright piano was
available , which could provide home entertainment while saving space . Types This section does not cite any sources . Please help improve this section by
adding citations to reliable sources .
</item 9>

<item 10>
Bed
Silk , velvet , and even cloth of gold were frequently used . Inventories from the beginning of the 14th century give details of these hangings lined with fur
and richly embroidered . It was then that the Four poster bed ( also known as a tester bed ) made its first appearance , the bed being slung from the ceiling or
fastened to the walls , a form which developed later into a room within a room , shut in by double curtains , sometimes even to exclude all drafts . The space
between bed and wall was called the ruelle , and very intimate friends were received there . The 14th century is also the time when feather beds became highly
prized possessions . In the 15th century beds became very large , reaching 7 to 8 feet ( 2.1 to 2.4 m ) by 6 to 7 feet ( 1.8 to 2.1 m ) . The mattresses were
often filled with pea - shucks , straw , or feathers .
</item 10>
=== END ITEM id=d087ba366388d787 ===

=== ITEM id=036349fed3870aa2 ===
Gold sentences and facts:
Sentence 0: Barbara Eden as Stella Johnson , a widowed cosmetics salesperson with a young daughter , whose joie de vivre , her principles , and her individuality anger most of the powerful people in town and on the Parent Teachers Association board . She inevitably becomes PTA president after embarrassing Flora , the snobby former President and her equally snobby friends . Ronny Cox as Willis `` Will '' Newton , a wealthy member of the PTA who does n't agree with Flora and her snobby allies and is in love with Stella . He and Stella get married at the end of the movie . Nanette Fabray as Alice Finley , Stella 's best friend and co-conspirator against the snobs of Harper Valley . She owns and operates the La Moderne Beauty Shop . Louis Nye as Kirby Baker , a local real estate agent who is allied with Flora and her friends . Notorious for being a lecherous playboy and infamous for impregnating his secretaries , one of whom , Gladys Wilmot , left town to see an abortionist in Cincinnati . As part of a plan to discredit Stella , he tries to foreclose on Stella 's house , which infuriates her . He then gets arrested and jailed for attacking Myrna Wong , a friend of Stella 's , who rebuffs his advances , throwing him by use of martial arts and destroying the office . John Fiedler as Bobby Taylor , a man on the PTA who lusts after Stella . Overly dominated by his wife , Holly . Although open to allow Stella to speak , he is not above having a fellow PTA board member kidnapped to commit election fraud . He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others ) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main antagonist . She is the pompous and extremely snobbish president of the PTA board who thoroughly despises Stella ( who defeats her for president at the end of the movie ) and all she stands for . So desperate is she to retain her power and prestige in the community that she has a fellow member of the PTA ( Skeeter Duggan ) kidnapped so she and her allies can commit election fraud . She is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon . Molly Dodd as Olive Glover , the recording secretary of the PTA who is a gambling addict . Clearly the most corrupt and most vindictive member of the PTA board , she is eventually arrested for embezzlement after having a load of hay and horse manure dumped on top of her . She is the one who initially angers Stella by sending her the letter denouncing her , and tries to get Dee 's friend , Mavis , arrested for embezzlement . Fay DeWitt as Willamae Jones , a snide biology teacher at the junior high school who is also a member of the PTA . She makes no secret of the fact that she favors the Reilly twins , due to the connections between her and Flora . Notorious for being a nymphomaniac ( though she hides it under a prim - and - proper façade ) , she is caught in flagrante delicto with Barney Crunk , the town 's mailman , on film . DeVera Marcus as Holly Taylor , Bobby 's obviously overbearing wife , who tends to drink a lot when Bobby is out of town , and does n't approve of Bobby 's lusting , or of anything else he does ( this is different from the character in the song , who `` seems to use a lot of ice whenever ( her husband ) 's away '' , implying adultery with the iceman ) . However , despite her ties to Flora and her scheming friends , she , like Shirley Thompson , is spared any retribution from Stella , due to Stella refusing to date Bobby . Louise Foley as Mavis , Dee Johnson 's best friend , who is nearly framed for a crime she did not commit by the vindictive Olive Glover . Brian Cook as Carlyle , a popular track star who chooses Dee , after she gets a makeover , and angers Bettina Reilly in the process . Laura Teige as Bettina Reilly , One of Flora 's twin granddaughters , who wants to be with Carlyle , and is jealous when he chooses Dee over her . She tends to be as snotty as her grandmother . Jan Teige as Edwina Reilly , Bettina 's twin sister , who is just as snotty as her . Irene Yah - Ling Sun as Myrna Wong , a friend of Stella 's who is a judo expert ; she is assaulted by Kirby Baker ( after leading him on ; she destroys him and his office in retaliation ) and has him sent to jail for assault . Tobias Anderson as `` Cranky '' Barney Crunk , the town 's nosy mailman who had the bad habit of peeking in everyone 's mail . He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson - Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her house is destroyed by pink - painted elephants .
  Fact 0.0: Barbara Eden plays Stella Johnson.
  Fact 0.1: Stella Johnson is a widowed cosmetics salesperson.
  Fact 0.2: Stella Johnson has a young daughter.
  Fact 0.3: Stella Johnson's joie de vivre angers most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.4: Stella Johnson's principles anger most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.5: Stella Johnson's individuality angers most of the powerful people in town and on the Parent Teachers Association board.
  Fact 0.6: Stella Johnson inevitably becomes PTA president after embarrassing Flora, the snobby former President.
  Fact 0.7: Stella Johnson inevitably becomes PTA president after embarrassing Flora's equally snobby friends.
  Fact 0.8: Ronny Cox plays Willis 'Will' Newton.
  Fact 0.9: Will Newton is a wealthy member of the PTA.
  Fact 0.10: Will Newton does not agree with Flora.
  Fact 0.11: Will Newton does not agree with Flora's snobby allies.
  Fact 0.12: Will Newton is in love with Stella.
  Fact 0.13: Will Newton and Stella get married at the end of the movie.
  Fact 0.14: Nanette Fabray plays Alice Finley.
  Fact 0.15: Alice Finley is Stella's best friend.
  Fact 0.16: Alice Finley is Stella's co-conspirator against the snobs of Harper Valley.
  Fact 0.17: Alice Finley owns the La Moderne Beauty Shop.
  Fact 0.18: Alice Finley operates the La Moderne Beauty Shop.
  Fact 0.19: Louis Nye plays Kirby Baker.
  Fact 0.20: Kirby Baker is a local real estate agent.
  Fact 0.21: Kirby Baker is allied with Flora and her friends.
  Fact 0.22: Kirby Baker is notorious for being a lecherous playboy.
  Fact 0.23: Kirby Baker is infamous for impregnating his secretaries.
  Fact 0.24: Gladys Wilmot is one of Kirby Baker's secretaries.
  Fact 0.25: Kirby Baker impregnated Gladys Wilmot.
  Fact 0.26: Gladys Wilmot left town to see an abortionist in Cincinnati.
  Fact 0.27: As part of a plan to discredit Stella, Kirby Baker tries to foreclose on Stella's house.
  Fact 0.28: Stella is infuriated by Kirby Baker's attempt to foreclose on her house.
  Fact 0.29: Kirby Baker gets arrested and jailed for attacking Myrna Wong.
  Fact 0.30: Myrna Wong is a friend of Stella's.
  Fact 0.31: Myrna Wong rebuffs Kirby Baker's advances.
  Fact 0.32: Myrna Wong throws Kirby Baker by use of martial arts.
  Fact 0.33: Myrna Wong destroys Kirby Baker's office.
  Fact 0.34: John Fiedler plays Bobby Taylor.
  Fact 0.35: Bobby Taylor is a man on the PTA.
  Fact 0.36: Bobby Taylor lusts after Stella.
  Fact 0.37: Bobby Taylor is overly dominated by his wife, Holly.
  Fact 0.38: Bobby Taylor is open to allowing Stella to speak.
  Fact 0.39: Bobby Taylor is not above having a fellow PTA board member kidnapped to commit election fraud.
  Fact 0.40: Bobby Taylor was arrested for indecent exposure.
  Fact 0.41: Bobby Taylor was set up by Stella.
  Fact 0.42: Ron Masak plays Herbie Maddox.
  Fact 0.43: Herbie Maddox is the head bartender at Kelly's bar.
  Fact 0.44: Herbie Maddox is Stella's ally against Flora and her friends.
  Fact 0.45: Herbie Maddox has a crush on Stella.
  Fact 0.46: Herbie Maddox is one of Stella's friends.
  Fact 0.47: Clint Howard plays Corley.
  Fact 0.48: Corley is a young delivery person.
  Fact 0.49: Corley works for the local florist shop.
  Fact 0.50: Corley brings Stella flowers from Will.
  Fact 0.51: Susan Swift plays Dee Johnson.
  Fact 0.52: Dee Johnson is Stella's 14-year-old daughter.
  Fact 0.53: Dee Johnson is threatened with expulsion from school if her mother does not conform to the way the PTA wants her to behave.
  Fact 0.54: Dee Johnson has a hand in assisting her mother and Alice with one of their revenge schemes.
  Fact 0.55: Dee Johnson loses her braces.
  Fact 0.56: Dee Johnson gets a makeover.
  Fact 0.57: Dee Johnson ends up with a boyfriend, Carlyle.
  Fact 0.58: Pat Paulsen plays Otis Harper.
  Fact 0.59: Otis Harper is the descendant of the founding family of Harper Valley.
  Fact 0.60: Otis Harper is a PTA member allied with Flora and her friends.
  Fact 0.61: Otis Harper has a tendency to drink all the time.
  Fact 0.62: Otis Harper's drinking annoys his wife, Bertha.
  Fact 0.63: Otis Harper had a stampede of pink-painted elephants storm through his house.
  Fact 0.64: The stampede of pink-painted elephants destroyed Otis Harper's house.
  Fact 0.65: Amzie Strickland plays Shirley Thompson.
  Fact 0.66: Shirley Thompson is a member of the PTA board.
  Fact 0.67: Shirley Thompson tends to drink a lot of gin.
  Fact 0.68: Shirley Thompson is allied with Flora and her friends.
  Fact 0.69: Shirley Thompson is one of four PTA members spared any retribution from Stella.
  Fact 0.70: The other three PTA members spared any retribution from Stella are Will, Skeeter and Holly Taylor.
  Fact 0.71: Shirley Thompson was seen as the most sympathetic member of the board.
  Fact 0.72: Shirley Thompson tended to be more manipulated than manipulative.
  Fact 0.73: It was never revealed whether Shirley Thompson retained her seat on the PTA.
  Fact 0.74: It is presumed that Shirley Thompson became more sympathetic towards Stella.
  Fact 0.75: Shirley Thompson was seen waving goodbye at Olive as Olive was being arrested.
  Fact 0.76: Bob Hastings plays Skeeter Duggan.
  Fact 0.77: Skeeter Duggan is a member of the PTA board.
  Fact 0.78: Skeeter Duggan is the town's Notary public.
  Fact 0.79: Skeeter Duggan is sympathetic to Stella.
  Fact 0.80: Dutch and Tex kidnap Skeeter Duggan at Flora Simpson-Reilly's orders.
  Fact 0.81: Audrey Christie plays Flora Simpson-Reilly.
  Fact 0.82: Flora Simpson-Reilly is the movie's main antagonist.
  Fact 0.83: Flora Simpson-Reilly is the pompous and extremely snobbish president of the PTA board.
  Fact 0.84: Flora Simpson-Reilly thoroughly despises Stella.
  Fact 0.85: Stella defeats Flora Simpson-Reilly for president at the end of the movie.
  Fact 0.86: Flora Simpson-Reilly despises all that Stella stands for.
  Fact 0.87: Flora Simpson-Reilly is desperate to retain her power and prestige in the community.
  Fact 0.88: Flora Simpson-Reilly has a fellow PTA member, Skeeter Duggan, kidnapped so she and her allies can commit election fraud.
  Fact 0.89: Flora Simpson-Reilly is embarrassed when her permanent set is destroyed by a hair depilatory at the beauty salon.
  Fact 0.90: Molly Dodd plays Olive Glover.
  Fact 0.91: Olive Glover is the recording secretary of the PTA.
  Fact 0.92: Olive Glover is a gambling addict.
  Fact 0.93: Olive Glover is clearly the most corrupt and most vindictive member of the PTA board.
  Fact 0.94: Olive Glover is eventually arrested for embezzlement after having a load of hay and horse manure dumped on top of her.
  Fact 0.95: Olive Glover initially angers Stella by sending her the letter denouncing her.
  Fact 0.96: Olive Glover tries to get Dee's friend, Mavis, arrested for embezzlement.
  Fact 0.97: Fay DeWitt plays Willamae Jones.
  Fact 0.98: Willamae Jones is a snide biology teacher at the junior high school.
  Fact 0.99: Willamae Jones is a member of the PTA.
  Fact 0.100: Willamae Jones favors the Reilly twins.
  Fact 0.101: Willamae Jones favors the Reilly twins due to the connections between her and Flora.
  Fact 0.102: Willamae Jones is notorious for being a nymphomaniac.
  Fact 0.103: Willamae Jones hides being a nymphomaniac under a prim-and-proper facade.
  Fact 0.104: Willamae Jones is caught in flagrante delicto with Barney Crunk, the town's mailman, on film.
  Fact 0.105: DeVera Marcus plays Holly Taylor.
  Fact 0.106: Holly Taylor is Bobby's obviously overbearing wife.
  Fact 0.107: Holly Taylor tends to drink a lot when Bobby is out of town.
  Fact 0.108: Holly Taylor does not approve of Bobby's lusting.
  Fact 0.109: Holly Taylor does not approve of anything else Bobby does.
  Fact 0.110: The Holly Taylor character differs from the character in the song, who seems to use a lot of ice whenever her husband is away, implying adultery with the iceman.
  Fact 0.111: Holly Taylor has ties to Flora and her scheming friends.
  Fact 0.112: Holly Taylor, like Shirley Thompson, is spared any retribution from Stella.
  Fact 0.113: Stella spares Holly Taylor due to Stella refusing to date Bobby.
  Fact 0.114: Louise Foley plays Mavis.
  Fact 0.115: Mavis is Dee Johnson's best friend.
  Fact 0.116: Mavis is nearly framed for a crime she did not commit by the vindictive Olive Glover.
  Fact 0.117: Brian Cook plays Carlyle.
  Fact 0.118: Carlyle is a popular track star.
  Fact 0.119: Carlyle chooses Dee after she gets a makeover.
  Fact 0.120: Carlyle's choice of Dee angers Bettina Reilly in the process.
  Fact 0.121: Laura Teige plays Bettina Reilly.
  Fact 0.122: Bettina Reilly is one of Flora's twin granddaughters.
  Fact 0.123: Bettina Reilly wants to be with Carlyle.
  Fact 0.124: Bettina Reilly is jealous when Carlyle chooses Dee over her.
  Fact 0.125: Bettina Reilly tends to be as snotty as her grandmother.
  Fact 0.126: Jan Teige plays Edwina Reilly.
  Fact 0.127: Edwina Reilly is Bettina's twin sister.
  Fact 0.128: Edwina Reilly is just as snotty as Bettina.
  Fact 0.129: Irene Yah-Ling Sun plays Myrna Wong.
  Fact 0.130: Myrna Wong is a friend of Stella's.
  Fact 0.131: Myrna Wong is a judo expert.
  Fact 0.132: Myrna Wong is assaulted by Kirby Baker after she leads him on.
  Fact 0.133: Myrna Wong destroys Kirby Baker and his office in retaliation.
  Fact 0.134: Myrna Wong has Kirby Baker sent to jail for assault.
  Fact 0.135: Tobias Anderson plays 'Cranky' Barney Crunk.
  Fact 0.136: Barney Crunk is the town's nosy mailman.
  Fact 0.137: Barney Crunk had the bad habit of peeking in everyone's mail.
  Fact 0.138: Barney Crunk is caught in flagrante delicto on film with Willamae Jones.
  Fact 0.139: Royce D. Applegate plays Dutch.
  Fact 0.140: Dutch is one of two kidnappers who abduct Skeeter Duggan at Flora Simpson-Reilly's orders.
  Fact 0.141: J.J. Barry plays Tex.
  Fact 0.142: Tex is the mastermind of the kidnap plot to get Skeeter Duggan out of the way.
  Fact 0.143: The kidnap plot lets Flora Simpson-Reilly and her friends commit election fraud.
  Fact 0.144: Tex is arrested.
  Fact 0.145: Dutch is arrested.
  Fact 0.146: Tex and Dutch rat out Flora and her friends on their scheme.
  Fact 0.147: Pitt Herbert plays Henry Reilly.
  Fact 0.148: Henry Reilly is Flora's overly dominated husband.
  Fact 0.149: Henry Reilly is the grandfather of Edwina and Bettina.
  Fact 0.150: Henry Reilly is wealthy.
  Fact 0.151: Henry Reilly is very weak-willed.
  Fact 0.152: Henry Reilly is known for acquiescing to his wife's schemes.
  Fact 0.153: Arlen Stuart plays Bertha Harper.
  Fact 0.154: Bertha Harper is the wife of Otis Harper.
  Fact 0.155: Bertha Harper gets annoyed by her husband's continual drinking.
  Fact 0.156: Bertha Harper is terrified when her house is destroyed by pink-painted elephants.

Retrieved items:
<item 1>
Harper Valley PTA (film)
He is caught in flagrante delicto on film with Willamae Jones . Royce D. Applegate as Dutch , one of two kidnappers who abduct Skeeter Duggan at Flora Simpson -
Reilly 's orders . J.J. Barry as Tex , the mastermind of the kidnap plot to get Skeeter Duggan out of the way so Flora Simpson - Reilly and her friends could
commit election fraud . He and his accomplice Dutch are arrested and the two of them rat out Flora and her friends on their scheme . Pitt Herbert as Henry
Reilly , Flora 's overly dominated husband and grandfather to Edwina and Bettina . Henry is wealthy , but is very weak - willed and is known for acquiescing to
his wife 's schemes . Arlen Stuart as Bertha Harper , wife of Otis Harper , who gets annoyed by her husband 's continual drinking . She is terrified when her
house is destroyed by pink - painted elephants .
</item 1>

<item 2>
Harper Valley PTA (film)
Harper Valley PTA ( film ) - wikipedia Harper Valley PTA ( film ) Jump to : navigation , search Harper Valley PTA Theatrical release poster Directed by Richard
Bennett Produced by George Edwards Written by George Edwards Barry Schneider Starring Barbara Eden Nanette Fabray Ronny Cox Louis Nye Susan Swift Pat Paulsen
Music by Nelson Riddle Cinematography Willy Kurant Edited by Michael Economou Distributed by April Fools Productions Release date May 23 , 1978 ( 1978 - 05 - 23
) ( Midwest ) June 2 , 1978 ( 1978 - 06 - 02 ) ( Canada ) August 2 , 1978 ( 1978 - 08 - 02 ) ( Los Angeles ) January 12 , 1979 ( 1979 - 01 - 12 ) ( New York
City ) Running time 93 minutes Country United States Language English Budget $1 million Box office $25 million ( USA ) Harper Valley PTA is a 1978 American
comedy film inspired by the popular 1968 country song `` Harper Valley PTA '' written by Tom T. Hall and performed by country singer Jeannie C. Riley .
</item 2>

<item 3>
Harper Valley PTA (film)
Side 1 `` Harper Valley PTA '' -- performed by Jeannie C. Riley `` Dee 's Visit '' ( instrumental ) -- performed by Nelson Riddle `` Mr. Harper '' -- performed
by Barbara Eden `` Alice 's Place '' ( instrumental ) -- performed by Nelson Riddle `` High School Confidential '' -- performed by Jerry Lee Lewis `` Willie Mae
'' ( instrumental ) -- performed by Nelson Riddle Side 2 `` Harper Valley PTA '' ( instrumental ) -- performed by Nelson Riddle `` Widow Jones '' -- performed
by Barbara Eden `` Twin Tune '' ( instrumental ) -- performed by Nelson Riddle `` Ballad of a Teenage Queen '' -- performed by Johnny Cash `` Ice Cream Disco ''
( instrumental ) -- performed by Nelson Riddle `` Whatever Happened to Charlie Brown '' -- performed by Rita Remington & Carol Channing `` Ending ( Reprise :
Harper Valley PTA ) '' ( instrumental ) -- performed by Nelson Riddle Television series ( edit ) Main article : Harper Valley PTA ( TV series ) In 1981 , Harper
Valley PTA was made into a television sitcom ( created by Sherwood Schwartz ) which aired on NBC from January 1981 to May 1982 .
</item 3>

<item 4>
Harper Valley PTA (film)
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
Harper Valley PTA (film)
The film grossed over $2 million its opening weekend , $5.8 million in its first three weeks and over $16 million after twelve weeks . The film opened in Los
Angeles on August 2 , 1978 and in New York City on January 12 , 1979 . Soundtrack ( edit ) Harper Valley PTA : Original Soundtrack Recording Soundtrack album by
various artists Released June 1978 Recorded Singleton Sound Studios Nashville , Tennessee Genre Soundtrack Length 29 : 50 Label Plantation Records Producer
Shelby S. Singleton , Jr . Singles from Harper Valley PTA `` Harper Valley PTA ( re-issue ) '' Released : July 1978 In June 1978 , a soundtrack album was
released on vinyl , cassette and 8 - track tape by Plantation Records . To promote the film 's release and its soundtrack , the title song by Jeannie C. Riley
was re-issued as a single . The album made its compact disc premiere through Varèse Sarabande on October 27 , 2017 .
</item 5>

<item 6>
Harper Valley PTA (film)
She is terrified when her house is destroyed by pink - painted elephants . Production and release ( edit ) Filming ( edit ) Harper Valley PTA was filmed in
twenty - seven days from October 3 , 1977 to December 8 , 1977 . It was filmed on location for one week in the town of Lebanon , Ohio and then continued in Los
Angeles , California . The track meet scene was filmed at Simi Valley High School On October 31 , 1977 , production ceased temporarily when Nanette Fabray was
knocked to the ground by a spooked elephant while shooting a scene involving pink elephants in the Hancock Park neighborhood of Los Angeles , California ; she
suffered a severe concussion , bruises and back sprains and was taken to Cedars - Sinai Medical Center where her condition was reported as serious but stable .
Following her recovery , filming resumed on November 30 . Release ( edit ) Harper Valley PTA opened in six theaters in Lebanon , Cincinnati and Dayton , Ohio on
May 23 , 1978 in accordance with executive producer Phil Borack 's plan to release the film first in smaller markets , where regional success could encourage
exhibitors in bigger cities to book the film .
</item 6>

<item 7>
Harper Valley PTA (film)
He had a stampede of elephants , painted pink , stormed through his house , destroying it . Amzie Strickland as Shirley Thompson , a member of the PTA board who
tends to drink a lot of gin . Though she is allied with Flora and her friends , she is one of four PTA members ( Will , Skeeter and Holly Taylor are the others
) who are spared any retribution from Stella . She was seen as the most sympathetic of the board , as she tended to be more manipulated as opposed to being
manipulative . It was never revealed whether or not she retained her seat on the PTA , but it is presumed that she became more sympathetic towards Stella as she
was seen waving goodbye at Olive as she was being arrested . Bob Hastings as Skeeter Duggan , a member of the PTA board , and the town 's Notary public who is
sympathetic to Stella . He is kidnapped by Dutch and Tex at Flora Simpson Reilly 's orders . Audrey Christie as Flora Simpson - Reilly , The movie 's main
antagonist .
</item 7>

<item 8>
Harper Valley PTA (film)
He and Stella get married at the end of the movie . Nanette Fabray as Alice Finley , Stella 's best friend and co-conspirator against the snobs of Harper Valley
. She owns and operates the La Moderne Beauty Shop . Louis Nye as Kirby Baker , a local real estate agent who is allied with Flora and her friends . Notorious
for being a lecherous playboy and infamous for impregnating his secretaries , one of whom , Gladys Wilmot , left town to see an abortionist in Cincinnati . As
part of a plan to discredit Stella , he tries to foreclose on Stella 's house , which infuriates her . He then gets arrested and jailed for attacking Myrna Wong
, a friend of Stella 's , who rebuffs his advances , throwing him by use of martial arts and destroying the office . John Fiedler as Bobby Taylor , a man on the
PTA who lusts after Stella . Overly dominated by his wife , Holly . Although open to allow Stella to speak , he is not above having a fellow PTA board member
kidnapped to commit election fraud .
</item 8>

<item 9>
Harper Valley PTA (film)
He was arrested for indecent exposure after being set up by Stella . Ron Masak as Herbie Maddox , the head bartender at Kelly 's bar , and Stella 's ally
against Flora and her friends . He has a crush on Stella , and is one of her friends . Clint Howard as Corley , a young delivery person , who works for the
local florist shop , who brings Stella flowers from Will . Susan Swift as Dee Johnson , Stella 's 14 - year - old daughter , who is threatened with expulsion
from school if her mother does n't conform to the way the PTA wants her to behave . She has a hand in assisting her mother and Alice with one of their revenge
schemes . She loses her braces , gets a makeover , and ends up with a boyfriend , Carlyle . Pat Paulsen as Otis Harper , the descendant of the founding family
of Harper Valley . A PTA member who is allied with Flora and her friends . Has a tendency to drink all the time , which annoys his wife , Bertha . He had a
stampede of elephants , painted pink , stormed through his house , destroying it .
</item 9>

<item 10>
Harper Valley PTA (film)
Barbara Eden reprised her role as Stella Johnson for the series which lasted two seasons and a total of 30 episodes were produced . References ( edit ) Jump up
^ St. Petersburg Times Jump up ^ Milwaukee Sentinel : Elephant injures Nanette Fabray Jump up ^ Observer Reporter External links ( edit ) Harper Valley PTA on
IMDb Harper Valley PTA at the TCM Movie Database Harper Valley PTA at AllMovie Harper Valley PTA at the American Film Institute Catalog Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Harper_Valley_PTA_(film)&oldid=818590841 '' Categories : 1978 films English - language films 1970s comedy films
American comedy films American films American independent films Feminist films Films about revenge Films adapted into television programs Films based on songs
Films set in Ohio Films shot in Ohio Films shot in Los Angeles 1978 soundtracks Film soundtracks Hidden categories : Articles with hAudio microformats Album
infoboxes lacking a cover Music infoboxes with deprecated parameters Talk Contents About Wikipedia Add links This page was last edited on 4 January 2018 , at 12
: 58 .
</item 10>
=== END ITEM id=036349fed3870aa2 ===

=== ITEM id=49691e33c9b2eeec ===
Gold sentences and facts:
Sentence 0: DJ Lance Rock -- ( Portrayed by Lance Robertson U.S. , voice dubbed by Ortis Deley in the UK ) A DJ who is the narrator and friend of the other characters and the series ' host . Muno -- ( Voiced by Adam Deibert U.S. , Bob Golding UK ) A friendly red cyclops boy . He 's the tallest , is somewhat clumsy and has a close bond with Foofa ( though she also bonds with Brobee in the same way ) . He is also the band 's guitarist . In the episodes `` Family '' , `` Circus '' , and `` Baby '' , his family makes an appearance . During Super Bowl XLIV , he appeared in a commercial for the Kia Sorento . His realm resembles the moon , another planet , or a desert and he has a pet horse . He has three stomachs as seen in the X-ray in the Season 1 episode `` Halloween '' ( # 13 ) . However another X-ray in the Season 1 episode `` Imagine '' ( # 20 ) shows him with one stomach , a normal heart and lungs . His catchphrase is `` Razzle Dazzle '' . Foofa -- ( Voiced by Emma Jacobs - Briggs U.S. , Teresa Gallagher UK ) A female flower bubble character who 's `` pink and happy . '' She likes flowers , rainbows and unicorns . Her instrument is the tambourine . Her realm is a spring or summer meadow . Brobee -- ( Voiced by Amos Watene U.S. , Simon Feilder UK ) A small , green monster boy and the baby of the gabbas . He is the drummer for the Gabba band . He is the one of the characters whose facial expression changes depending on the situation ; his normal face turning into a frown , and vice versa . He is often subjected to trying new foods . His realm is a vast autumn forest . It is mentioned that he is four years old . Toodee -- ( Voiced by Erin Pearce U.S. , Jonell Elliot UK ) A blue arctic female cat - dragon who loves to have fun . She has a close bond with Plex and is the bassist for the Gabba band . She resides in a winter arctic realm , with icicles . Plex -- ( Voiced by Christian Jacobs U.S. , Simon Feilder UK ) A magic yellow robot boy and the oldest of the gabbas and he is Toodee 's love instest . He is portrayed as being smart and as the leader of Gabba Land . Plex usually teaches lessons to his friends and is a father figure to the gang . He can also use his special ray to bring things into existence . His instrument is the keytar . He has no realm in particular , though he seems to have a `` docking station '' between Foofa and Brobee 's realms . In `` Gabbaland Begins '' , Plex is the last in DJ Lance 's original creations after he goes through Muno , Foofa , Brobee , and Toodee , although , Brobee is supposedly a four - year - old in the series . Gooble - ( Voiced by Joel Fox U.S. ) A ghost who is always so sad and crying . He resides in a haunted house located in a non-gabba land - place named `` Gooble Land '' .
  Fact 0.0: DJ Lance Rock is portrayed by Lance Robertson in the U.S.
  Fact 0.1: DJ Lance Rock's voice is dubbed by Ortis Deley in the UK.
  Fact 0.2: DJ Lance Rock is a DJ.
  Fact 0.3: DJ Lance Rock is the narrator of the series.
  Fact 0.4: DJ Lance Rock is a friend of the other characters.
  Fact 0.5: DJ Lance Rock is the host of the series.
  Fact 0.6: Muno is voiced by Adam Deibert in the U.S.
  Fact 0.7: Muno is voiced by Bob Golding in the UK.
  Fact 0.8: Muno is a friendly red cyclops boy.
  Fact 0.9: Muno is the tallest.
  Fact 0.10: Muno is somewhat clumsy.
  Fact 0.11: Muno has a close bond with Foofa.
  Fact 0.12: Foofa also bonds with Brobee in the same way.
  Fact 0.13: Muno is also the band's guitarist.
  Fact 0.14: Muno's family makes an appearance in the episodes 'Family', 'Circus', and 'Baby'.
  Fact 0.15: Muno appeared in a commercial for the Kia Sorento during Super Bowl XLIV.
  Fact 0.16: Muno's realm resembles the moon, another planet, or a desert.
  Fact 0.17: Muno has a pet horse.
  Fact 0.18: Muno has three stomachs as seen in the X-ray in the Season 1 episode 'Halloween' (#13).
  Fact 0.19: Another X-ray in the Season 1 episode 'Imagine' (#20) shows Muno with one stomach.
  Fact 0.20: Another X-ray in the Season 1 episode 'Imagine' (#20) shows Muno with a normal heart.
  Fact 0.21: Another X-ray in the Season 1 episode 'Imagine' (#20) shows Muno with normal lungs.
  Fact 0.22: Muno's catchphrase is 'Razzle Dazzle'.
  Fact 0.23: Foofa is voiced by Emma Jacobs-Briggs in the U.S.
  Fact 0.24: Foofa is voiced by Teresa Gallagher in the UK.
  Fact 0.25: Foofa is a female flower bubble character.
  Fact 0.26: Foofa is pink and happy.
  Fact 0.27: Foofa likes flowers, rainbows and unicorns.
  Fact 0.28: Foofa's instrument is the tambourine.
  Fact 0.29: Foofa's realm is a spring or summer meadow.
  Fact 0.30: Brobee is voiced by Amos Watene in the U.S.
  Fact 0.31: Brobee is voiced by Simon Feilder in the UK.
  Fact 0.32: Brobee is a small, green monster boy.
  Fact 0.33: Brobee is the baby of the gabbas.
  Fact 0.34: Brobee is the drummer for the Gabba band.
  Fact 0.35: Brobee is one of the characters whose facial expression changes depending on the situation.
  Fact 0.36: Brobee's normal face turns into a frown, and vice versa.
  Fact 0.37: Brobee is often subjected to trying new foods.
  Fact 0.38: Brobee's realm is a vast autumn forest.
  Fact 0.39: It is mentioned that Brobee is four years old.
  Fact 0.40: Toodee is voiced by Erin Pearce in the U.S.
  Fact 0.41: Toodee is voiced by Jonell Elliot in the UK.
  Fact 0.42: Toodee is a blue arctic female cat-dragon.
  Fact 0.43: Toodee loves to have fun.
  Fact 0.44: Toodee has a close bond with Plex.
  Fact 0.45: Toodee is the bassist for the Gabba band.
  Fact 0.46: Toodee resides in a winter arctic realm with icicles.
  Fact 0.47: Plex is voiced by Christian Jacobs in the U.S.
  Fact 0.48: Plex is voiced by Simon Feilder in the UK.
  Fact 0.49: Plex is a magic yellow robot boy.
  Fact 0.50: Plex is the oldest of the gabbas.
  Fact 0.51: Plex is Toodee's love interest.
  Fact 0.52: Plex is portrayed as smart.
  Fact 0.53: Plex is portrayed as the leader of Gabba Land.
  Fact 0.54: Plex usually teaches lessons to his friends.
  Fact 0.55: Plex is a father figure to the gang.
  Fact 0.56: Plex can use his special ray to bring things into existence.
  Fact 0.57: Plex's instrument is the keytar.
  Fact 0.58: Plex has no realm in particular.
  Fact 0.59: Plex seems to have a 'docking station' between Foofa's and Brobee's realms.
  Fact 0.60: In 'Gabbaland Begins', Plex is the last of DJ Lance's original creations.
  Fact 0.61: In 'Gabbaland Begins', Plex comes after Muno, Foofa, Brobee, and Toodee in DJ Lance's original creations.
  Fact 0.62: Brobee is supposedly a four-year-old in the series.
  Fact 0.63: Gooble is voiced by Joel Fox in the U.S.
  Fact 0.64: Gooble is a ghost.
  Fact 0.65: Gooble is always so sad and crying.
  Fact 0.66: Gooble resides in a haunted house.
  Fact 0.67: The haunted house is located in a non-Gabba land place named 'Gooble Land'.
Sentence 1: Hosted by a character named DJ Lance Rock , the series featured a mix of live - action segments featuring cartoonish costumed characters -- Muno ( the red cyclops ) , Foofa ( the pink flower bubble ) , Brobee ( a little hairy green monster ) , Toodee ( the blue cat - dragon ) and Plex ( the magic yellow robot ) -- and many short animated sketches and musical numbers .
  Fact 1.0: The series was hosted by a character named DJ Lance Rock.
  Fact 1.1: The series featured a mix of live-action segments.
  Fact 1.2: The live-action segments featured cartoonish costumed characters.
  Fact 1.3: The series featured many short animated sketches.
  Fact 1.4: The series featured many musical numbers.
  Fact 1.5: Muno is the red cyclops.
  Fact 1.6: Foofa is the pink flower bubble.
  Fact 1.7: Brobee is a little hairy green monster.
  Fact 1.8: Toodee is the blue cat-dragon.
  Fact 1.9: Plex is the magic yellow robot.

Retrieved items:
<item 1>
Graph query row:
character: Foofa
relationship: RELATED_TO
show: Yo Gabba Gabba!
</item 1>

<item 2>
Yo Gabba Gabba!
is an American live action / puppet educational children 's television show starring five costumed toys come to life and their friend DJ Lance Rock . The series
premiered on Nick Jr . August 20 , 2007 and ended on November 12 , 2015 . There is a single topic in each episode , e.g. `` Adventure '' , `` Friends '' , and
`` Dance '' , through songs and short storylines in the half - hour program . Additionally , the show teaches children life and social skills , such as sharing
and trying new foods . It also encourages viewers to move along with and dance with the characters in the program . The show is noted for its indie - culture
guest stars and bands , and for drawing visual inspiration from 8 - bit video games and H.R. Pufnstuf , among other television shows . Created by Christian
Jacobs ( lead singer of the Aquabats ) and Scott Schultz , the show is written to appeal to children and their parents . The television program has spawned a
touring live stage show , various toys and branded clothing . Trademarks relating to Yo Gabba Gabba !
</item 2>

<item 3>
Graph query row:
character: Plex
relationship: RELATED_TO
show: Yo Gabba Gabba!
</item 3>

<item 4>
Graph query row:
character: Toodee
relationship: RELATED_TO
show: Yo Gabba Gabba!
</item 4>

<item 5>
Graph query row:
character: DJ Lance Rock
relationship: RELATED_TO
show: Yo Gabba Gabba!
</item 5>

<item 6>
Graph query row:
character: Muno
relationship: RELATED_TO
show: Yo Gabba Gabba!
</item 6>

<item 7>
Yo Gabba Gabba!
Trademarks relating to Yo Gabba Gabba ! and its characters are held by GabbaCaDabra , LLC. in partnership with The Magic Store and WildBrain . Contents ( hide )
1 Overview 2 History 3 Production 4 Episode structure 5 The Stars 6 Regular segments 7 Episodes 8 Video releases 9 Soundtrack releases 10 Stage shows 11 Mobile
streaming 12 Critical reception 13 In other media 14 References 15 External links Overview ( edit ) Hosted by a character named DJ Lance Rock , the series
featured a mix of live - action segments featuring cartoonish costumed characters -- Muno ( the red cyclops ) , Foofa ( the pink flower bubble ) , Brobee ( a
little hairy green monster ) , Toodee ( the blue cat - dragon ) and Plex ( the magic yellow robot ) -- and many short animated sketches and musical numbers .
</item 7>

<item 8>
Yo Gabba Gabba!
Yo Gabba Gabba ! - Wikipedia Yo Gabba Gabba ! Jump to : navigation , search Yo Gabba Gabba ! Yo Gabba Gabba ! title card Genre Educational Created by Christian
Jacobs Scott Schultz Developed by Kay Wilson Stallings Directed by Christian Jacobs Scott Schultz Matt Fackrell Matt Chapman Presented by Lance Robertson Voices
of Adam Delbert Christian Jacobs Amos `` Mossi '' Watene Erin Pearce Emma Jacobs - Briggs Country of origin United States Original language ( s ) English No. of
seasons No. of episodes 66 ( list of episodes ) Production Executive producer ( s ) Michael Polis Jon Berrett Producer ( s ) Justin Lyon Ritamarie Peruggi
Running time 25 minutes Production company ( s ) The Magic Store W ! ldbrain Entertainment Nickelodeon Productions Distributor DHX Media Release Original
network Nickelodeon Nick Jr . Original release August 20 , 2007 ( 2007 - 08 - 20 ) -- November 12 , 2015 ( 2015 - 11 - 12 ) External links Website Yo Gabba
Gabba !
</item 8>

<item 9>
Yo Gabba Gabba!
The toy models of the characters that appeared at the beginning and end of each show were made by Kidrobot . History ( edit ) Yo Gabba Gabba ! was developed by
two Southern California fathers , Christian Jacobs and Scott Schultz , who first started working together as teenagers , producing and directing skateboarding
videos . Their goal was to design a kids ' show that was entertaining while featuring real artists and real performers . Both had no previous experience writing
for television , let alone children 's broadcasting or education . In developing the show they took inspiration from a number of sources including Sesame Street
, The Electric Company , Pee - wee 's Playhouse , as well as Sid and Marty Krofft puppet shows Banana Splits and H.R. Pufnstuf . In 1999 , after becoming
parents , Jacobs and Schultz started playing around with ideas for children 's television and produced a pilot independently financed by small loans from
friends and family . Yo Gabba Gabba ! did not get much attention until it started circulating on the Internet .
</item 9>

<item 10>
Graph query row:
character: Brobee
relationship: RELATED_TO
show: Yo Gabba Gabba!
</item 10>
=== END ITEM id=49691e33c9b2eeec ===

=== ITEM id=3d57363a894cd3fb ===
Gold sentences and facts:
Sentence 0: The small intestine or small bowel is the part of the gastrointestinal tract between the stomach and the large intestine , and is where most of the end absorption of food takes place . The small intestine has three distinct regions -- the duodenum , jejunum , and ileum . The duodenum is the shortest part of the small intestine and is where preparation for absorption begins . It also receives bile and pancreatic juice through the pancreatic duct , controlled by the sphincter of Oddi . The primary function of the small intestine is the absorption of nutrients and minerals from food , using small finger - like protrusions called villi .
  Fact 0.0: The small intestine is also called the small bowel.
  Fact 0.1: The small intestine is the part of the gastrointestinal tract between the stomach and the large intestine.
  Fact 0.2: Most of the end absorption of food takes place in the small intestine.
  Fact 0.3: The small intestine has three distinct regions.
  Fact 0.4: The three regions of the small intestine are the duodenum, jejunum, and ileum.
  Fact 0.5: The duodenum is the shortest part of the small intestine.
  Fact 0.6: Preparation for absorption begins in the duodenum.
  Fact 0.7: The duodenum receives bile through the pancreatic duct.
  Fact 0.8: The duodenum receives pancreatic juice through the pancreatic duct.
  Fact 0.9: The flow of bile and pancreatic juice into the duodenum is controlled by the sphincter of Oddi.
  Fact 0.10: The primary function of the small intestine is the absorption of nutrients and minerals from food.
  Fact 0.11: The small intestine uses small finger-like protrusions called villi for absorption.

Retrieved items:
<item 1>
Human digestive system
Most of the digestion of food takes place in the small intestine . Water and some minerals are reabsorbed back into the blood in the colon of the large
intestine . The waste products of digestion ( feces ) are defecated from the anus via the rectum . Contents ( hide ) 1 Components 1.1 Mouth 1.1. 1 Salivary
glands 1.1. 1.1 Saliva 1.1. 2 Tongue 1.1. 2.1 Taste 1.1. 3 Teeth 1.1. 4 Epiglottis 1.2 Pharynx 1.3 Esophagus 1.4 Diaphragm 1.5 Stomach 1.6 Spleen 1.7 Liver 1.7.
1 Bile 1.7. 2 Gallbladder 1.8 Pancreas 1.9 Lower gastrointestinal tract 1.9. 1 Small intestine 1.9. 2 Cecum 1.9.
</item 1>

<item 2>
Small intestine
This is because the cellulose is made out of beta - glucose , making the inter-monosaccharidal bindings different from the ones present in starch , which
consists of alpha - glucose . Humans lack the enzyme for splitting the beta - glucose - bonds , something reserved for herbivores and bacteria from the large
intestine . Absorption ( edit ) Digested food is now able to pass into the blood vessels in the wall of the intestine through either diffusion or active
transport . The small intestine is the site where most of the nutrients from ingested food are absorbed . The inner wall , or mucosa , of the small intestine is
lined with simple columnar epithelial tissue . Structurally , the mucosa is covered in wrinkles or folds called plicae circulares , which are considered
permanent features in the wall of the organ . They are distinct from rugae which are considered non-permanent or temporary allowing for distention and
contraction . From the plicae circulares project microscopic finger - like pieces of tissue called villi ( Latin for `` shaggy hair '' ) . The individual
epithelial cells also have finger - like projections known as microvilli .
</item 2>

<item 3>
Small intestine
The individual epithelial cells also have finger - like projections known as microvilli . The functions of the plicae circulares , the villi , and the
microvilli are to increase the amount of surface area available for the absorption of nutrients , and to limit the loss of said nutrients to intestinal fauna .
Each villus has a network of capillaries and fine lymphatic vessels called lacteals close to its surface . The epithelial cells of the villi transport nutrients
from the lumen of the intestine into these capillaries ( amino acids and carbohydrates ) and lacteals ( lipids ) . The absorbed substances are transported via
the blood vessels to different organs of the body where they are used to build complex substances such as the proteins required by our body . The material that
remains undigested and unabsorbed passes into the large intestine . Absorption of the majority of nutrients takes place in the jejunum , with the following
notable exceptions : Iron is absorbed in the duodenum . Vitamin B12 and bile salts are absorbed in the terminal ileum . Water is absorbed by osmosis and lipids
by passive diffusion throughout the small intestine .
</item 3>

<item 4>
Human digestive system
This breaks down fats to some degree though is not as efficient as the pancreatic lipase . The pylorus , the lowest section of the stomach which attaches to the
duodenum via the pyloric canal , contains countless glands which secrete digestive enzymes including gastrin . After an hour or two , a thick semi-liquid called
chyme is produced . When the pyloric sphincter , or valve opens , chyme enters the duodenum where it mixes further with digestive enzymes from the pancreas ,
and then passes through the small intestine , where digestion continues . When the chyme is fully digested , it is absorbed into the blood . 95 % of absorption
of nutrients occurs in the small intestine . Water and minerals are reabsorbed back into the blood in the colon of the large intestine , where the environment
is slightly acidic . Some vitamins , such as biotin and vitamin K produced by bacteria in the gut flora of the colon are also absorbed . The parietal cells in
the fundus of the stomach , produce a glycoprotein called intrinsic factor which is essential for the absorption of vitamin B12 .
</item 4>

<item 5>
Human digestive system
The celiac artery supplies the liver , stomach , spleen and the upper 1 / 3 of the duodenum ( to the sphincter of Oddi ) and the pancreas with oxygenated blood
. Most of the blood is returned to the liver via the portal venous system for further processing and detoxification before returning to the systemic circulation
via the hepatic portal vein . The next branch from the abdominal aorta is the superior mesenteric artery , which supplies the regions of the digestive tract
derived from the midgut , which includes the distal 2 / 3 of the duodenum , jejunum , ileum , cecum , appendix , ascending colon , and the proximal 2 / 3 of the
transverse colon . The final branch which is important for the digestive system is the inferior mesenteric artery , which supplies the regions of the digestive
tract derived from the hindgut , which includes the distal 1 / 3 of the transverse colon , descending colon , sigmoid colon , rectum , and the anus above the
pectinate line . Nerve supply ( edit ) Dietary life rules , Japan , Edo period .
</item 5>

<item 6>
Human digestive system
In the duodenum , pancreatic lipase is secreted together with a co-enzyme , colipase to further digest the fat content of the chyme . From this breakdown ,
smaller particles of emulsified fats called chylomicrons are produced . There are also digestive cells called enterocytes lining the intestines ( the majority
being in the small intestine ) . They are unusual cells in that they have villi on their surface which in turn have innumerable microvilli on their surface .
All these villi make for a greater surface area , not only for the absorption of chyme but also for its further digestion by large numbers of digestive enzymes
present on the microvilli . The chylomicrons are small enough to pass through the enterocyte villi and into their lymph capillaries called lacteals . A milky
fluid called chyle , consisting mainly of the emulsified fats of the chylomicrons , results from the absorbed mix with the lymph in the lacteals . Chyle is then
transported through the lymphatic system to the rest of the body .
</item 6>

<item 7>
Human digestive system
Chyle is then transported through the lymphatic system to the rest of the body . The suspensory muscle marks the end of the duodenum and the division between
the upper gastrointestinal tract and the lower GI tract . The digestive tract continues as the jejunum which continues as the ileum . The jejunum , the
midsection of the small intestine contains circular folds , flaps of doubled mucosal membrane which partially encircle and sometimes completely encircle the
lumen of the intestine . These folds together with villi serve to increase the surface area of the jejunum enabling an increased absorption of digested sugars ,
amino acids and fatty acids into the bloodstream . The circular folds also slow the passage of food giving more time for nutrients to be absorbed . The last
part of the small intestine is the ileum . This also contains villi and vitamin B12 ; bile acids and any residue nutrients are absorbed here . When the chyme is
exhausted of its nutrients the remaining waste material changes into the semi-solids called feces , which pass to the large intestine , where bacteria in the
gut flora further break down residual proteins and starches .
</item 7>

<item 8>
Small intestine
Water is absorbed by osmosis and lipids by passive diffusion throughout the small intestine . Sodium bicarbonate is absorbed by active transport and glucose and
amino acid co-transport Fructose is absorbed by facilitated diffusion . Immunological ( edit ) The small intestine supports the body 's immune system . The
presence of gut flora appears to contribute positively to the host 's immune system . Peyer 's patches , located within the ileum of the small intestine , are
an important part of the digestive tract 's local immune system . They are part of the lymphatic system , and provide a site for antigens from potentially
harmful bacteria or other microorganisms in the digestive tract to be sampled , and subsequently presented to the immune system . Gene and protein expression (
edit ) Further information : Bioinformatics § Gene and protein expression About 20,000 protein coding genes are expressed in human cells and 70 % of these genes
are expressed in the normal duodenum . Some 300 of these genes are more specifically expressed in the duodenum with very few genes expressed only in the small
intestine .
</item 8>

<item 9>
Small intestine
The three major classes of nutrients that undergo digestion are proteins , lipids ( fats ) and carbohydrates : Proteins are degraded into small peptides and
amino acids before absorption . Chemical breakdown begins in the stomach and continues in the small intestine . Proteolytic enzymes , including trypsin and
chymotrypsin , are secreted by the pancreas and cleave proteins into smaller peptides . Carboxypeptidase , which is a pancreatic brush border enzyme , splits
one amino acid at a time . Aminopeptidase and dipeptidase free the end amino acid products . Lipids ( fats ) are degraded into fatty acids and glycerol .
Pancreatic lipase breaks down triglycerides into free fatty acids and monoglycerides . Pancreatic lipase works with the help of the salts from the bile secreted
by the liver and stored in the gall bladder . Bile salts attach to triglycerides to help emulsify them , which aids access by pancreatic lipase .
</item 9>

<item 10>
Human digestive system
The mucus component lubricates the walls of the intestine . When the digested food particles are reduced enough in size and composition , they can be absorbed
by the intestinal wall and carried to the bloodstream . The first receptacle for this chyme is the duodenal bulb . From here it passes into the first of the
three sections of the small intestine , the duodenum . ( The next section is the jejunum and the third is the ileum ) . The duodenum is the first and shortest
section of the small intestine . It is a hollow , jointed C - shaped tube connecting the stomach to the jejunum . It starts at the duodenal bulb and ends at the
suspensory muscle of duodenum . The attachment of the suspensory muscle to the diaphragm is thought to help the passage of food by making a wider angle at its
attachment . Most food digestion takes place in the small intestine . Segmentation contractions act to mix and move the chyme more slowly in the small intestine
allowing more time for absorption ( and these continue in the large intestine ) .
</item 10>
=== END ITEM id=3d57363a894cd3fb ===

=== ITEM id=dc6f42faa475d782 ===
Gold sentences and facts:
Sentence 0: The right-most label conveys the top - level domain ; for example , the domain name www.example.com belongs to the top - level domain com . The hierarchy of domains descends from the right to the left label in the name ; each label to the left specifies a subdivision , or subdomain of the domain to the right . For example : the label example specifies a node example.com as a subdomain of the com domain , and www is a label to create www.example.com , a subdomain of example.com . Each label may contain from 1 to 63 octets . The empty label is reserved for the root node and when fully qualified is expressed as the empty label terminated by a dot . The full domain name may not exceed a total length of 253 ASCII characters in its textual representation . Thus , when using a single character per label , the limit is 127 levels : 127 characters plus 126 dots have a total length of 253 . In practice , some domain registries may have shorter limits . A hostname is a domain name that has at least one associated IP address . For example , the domain names www.example.com and example.com are also hostnames , whereas the com domain is not . However , other top - level domains , particularly country code top - level domains , may indeed have an IP address , and if so , they are also hostnames . Hostnames impose restrictions on the characters allowed in the corresponding domain name . A valid hostname is also a valid domain name , but a valid domain name may not necessarily be valid as a hostname .
  Fact 0.0: The right-most label of a domain name conveys the top-level domain.
  Fact 0.1: In www.example.com, the top-level domain is com.
  Fact 0.2: The hierarchy of domains descends from the right label to the left label in the name.
  Fact 0.3: Each label to the left specifies a subdivision, or subdomain, of the domain to its right.
  Fact 0.4: The label example specifies the node example.com as a subdomain of the com domain.
  Fact 0.5: The label www creates www.example.com, a subdomain of example.com.
  Fact 0.6: Each label may contain from 1 to 63 octets.
  Fact 0.7: The empty label is reserved for the root node.
  Fact 0.8: When fully qualified, the root node is expressed as the empty label terminated by a dot.
  Fact 0.9: The full domain name may not exceed a total length of 253 ASCII characters in its textual representation.
  Fact 0.10: When using a single character per label, the limit is 127 levels.
  Fact 0.11: 127 characters plus 126 dots have a total length of 253.
  Fact 0.12: In practice, some domain registries may have shorter limits.
  Fact 0.13: A hostname is a domain name that has at least one associated IP address.
  Fact 0.14: The domain names www.example.com and example.com are also hostnames.
  Fact 0.15: The com domain is not a hostname.
  Fact 0.16: Other top-level domains, particularly country code top-level domains, may have an IP address.
  Fact 0.17: If another top-level domain has an IP address, it is also a hostname.
  Fact 0.18: Hostnames impose restrictions on the characters allowed in the corresponding domain name.
  Fact 0.19: A valid hostname is also a valid domain name.
  Fact 0.20: A valid domain name may not necessarily be valid as a hostname.

Retrieved items:
<item 1>
Domain names are memorable, dot-separated DNS identifiers that map Internet resources to IP addresses. Their hierarchical labels run from TLDs through
second-level names and subdomains; hostnames have stricter rules. Domains support mobility, redundancy, and virtual hosting. ICANN coordinates DNS and accredits
registrars; registries maintain records, while registrants receive time-limited use—not ownership—and manage administrative, technical, and billing contacts.
TLDs include generic, country-code, special-use, and newer branded
</item 1>

<item 2>
Domain name
Below these levels , the next domain name component has been used to designate a particular host server . Therefore , ftp.example.com might be an FTP server ,
www.example.com would be a World Wide Web server , and mail.example.com could be an email server , each intended to perform only the implied function . Modern
technology allows multiple physical servers with either different ( cf . load balancing ) or even identical addresses ( cf . anycast ) to serve a single
hostname or domain name , or multiple domain names to be served by a single computer . The latter is very popular in Web hosting service centers , where service
providers host the websites of many organizations on just a few servers . The hierarchical DNS labels or components of domain names are separated in a fully
qualified name by the full stop ( dot , . ) . Internationalized domain names Main article : Internationalized domain name The character set allowed in the
Domain Name System is based on ASCII and does not allow the representation of names and words of many languages in their native scripts or alphabets . ICANN
approved the Internationalized domain name ( IDNA ) system , which maps Unicode strings used in application user interfaces into the valid DNS character set by
an encoding called Punycode .
</item 2>

<item 3>
Domain names consist of labels separated by dots, with each label forming a subdomain of the one to its right (for example, “sos” is under “state.oh.us”). DNS
names can have very deep hierarchies, as in IPv6 reverse-lookup zones. Second-level domains often identify a company, product, or service, while further labels
may identify a particular host server.
</item 3>

<item 4>
Domain name
Domain name - wikipedia Domain name Jump to : navigation , search This article is about domain names in the Internet . For other uses , see Domain . The
hierarchy of labels in a fully qualified domain name . A domain name is an identification string that defines a realm of administrative autonomy , authority or
control within the Internet . Domain names are formed by the rules and procedures of the Domain Name System ( DNS ) . Any name registered in the DNS is a domain
name . Domain names are used in various networking contexts and for application - specific naming and addressing purposes . In general , a domain name
represents an Internet Protocol ( IP ) resource , such as a personal computer used to access the Internet , a server computer hosting a web site , or the web
site itself or any other service communicated via the Internet . In 2017 , 330.6 million domain names had been registered . Domain names are organized in
subordinate levels ( subdomains ) of the DNS root domain , which is nameless .
</item 4>

<item 5>
A domain name is an identification string governed by the Domain Name System (DNS), defining an area of administrative authority or control on the Internet. It
can identify IP-based resources such as computers, servers, websites, or online services. Domain names form a hierarchy of subordinate levels, or subdomains,
beneath the nameless DNS root. In 2017, 330.6 million domain names were registered.
</item 5>

<item 6>
Domain names consist of dot-separated labels (1–63 octets); fully qualified names end at the root, and textual names are limited to 253 characters (up to 127
single-character labels). A hostname is a domain with an associated IP address and may face extra character restrictions. Domains can be valuable branding
assets; some registrars offset low or free registration costs with advertising or hosting requirements. Owners can create unlimited subdomains, though desirable
names may already be taken, prompting searches via registrars
</item 6>

<item 7>
Domain name
The first - level set of domain names are the top - level domains ( TLDs ) , including the generic top - level domains ( gTLDs ) , such as the prominent domains
com , info , net , edu , and org , and the country code top - level domains ( ccTLDs ) . Below these top - level domains in the DNS hierarchy are the second -
level and third - level domain names that are typically open for reservation by end - users who wish to connect local area networks to the Internet , create
other publicly accessible Internet resources or run web sites . The registration of these domain names is usually administered by domain name registrars who
sell their services to the public . A fully qualified domain name ( FQDN ) is a domain name that is completely specified with all labels in the hierarchy of the
DNS , having no parts omitted . Labels in the Domain Name System are case - insensitive , and may therefore be written in any desired capitalization method ,
but most commonly domain names are written in lowercase in technical contexts .
</item 7>

<item 8>
Domain name
Host names appear as a component in Uniform Resource Locators ( URLs ) for Internet resources such as web sites ( e.g. , en.wikipedia.org ) . Domain names are
also used as simple identification labels to indicate ownership or control of a resource . Such examples are the realm identifiers used in the Session
Initiation Protocol ( SIP ) , the Domain Keys used to verify DNS domains in e-mail systems , and in many other Uniform Resource Identifiers ( URIs ) . An
important function of domain names is to provide easily recognizable and memorizable names to numerically addressed Internet resources . This abstraction allows
any resource to be moved to a different physical location in the address topology of the network , globally or locally in an intranet . Such a move usually
requires changing the IP address of a resource and the corresponding translation of this IP address to and from its domain name . Domain names are used to
establish a unique identity . Organizations can choose a domain name that corresponds to their name , helping Internet users to reach them easily . A generic
domain is a name that defines a general category , rather than a specific or personal instance , for example , the name of an industry , rather than a company
name .
</item 8>

<item 9>
Domain name
Domain name space Today , the Internet Corporation for Assigned Names and Numbers ( ICANN ) manages the top - level development and architecture of the Internet
domain name space . It authorizes domain name registrars , through which domain names may be registered and reassigned . The hierarchical domain name system ,
organized into zones , each served by domain name servers . The domain name space consists of a tree of domain names . Each node in the tree holds information
associated with the domain name . The tree sub-divides into zones beginning at the DNS root zone . Domain name syntax A domain name consists of one or more
parts , technically called labels , that are conventionally concatenated , and delimited by dots , such as example.com . The right-most label conveys the top -
level domain ; for example , the domain name www.example.com belongs to the top - level domain com . The hierarchy of domains descends from the right to the
left label in the name ; each label to the left specifies a subdivision , or subdomain of the domain to the right .
</item 9>

<item 10>
Domain names are memorable text labels used to identify Internet resources—such as computers, networks, and services—instead of numerical protocol addresses. A
name can refer to a collection of resources or an individual instance. On a host computer, it serves as a host identifier or hostname; the term also refers to
leaf labels in the domain name system that have no subordinate domain space.
</item 10>
=== END ITEM id=dc6f42faa475d782 ===

=== ITEM id=bc4ab6e25bbbb224 ===
Gold sentences and facts:
Sentence 0: A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three vanishing points .
  Fact 0.0: A vanishing point is a point on the image plane of a perspective drawing.
  Fact 0.1: In a perspective drawing, the two-dimensional perspective projections, or drawings, of mutually parallel lines in three-dimensional space appear to converge at a vanishing point.
  Fact 0.2: When the set of parallel lines is perpendicular to a picture plane, the construction is known as one-point perspective.
  Fact 0.3: In one-point perspective, the vanishing point corresponds to the oculus, or "eye point", from which the image should be viewed for correct perspective geometry.
  Fact 0.4: Traditional linear drawings use objects with one to three sets of parallels.
  Fact 0.5: Traditional linear drawings define one to three vanishing points.

Retrieved items:
<item 1>
Vanishing point
Using perspective pinhole projections , a point on L projected on the image plane will have coordinates defined as , x ′ = f x / z = f d − bz / az y ′ = f y / z
= f c / z This is the parametric representation of the image L ′ of the line L with z as the parameter . When z → − ∞ it stops at the point ( x ′ , y ′ ) = ( −
fb / a , 0 ) on the x ′ axis of the image plane . This is the vanishing point corresponding to all parallel lines with slope − b / a in the plane π . All
vanishing points associated with different lines with different slopes belonging to plane π will lie on the x ′ axis , which in this case is the horizon line .
2 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x , y , f ) , v ≡ ( x , y , f ) be the three
corresponding vanishing points respectively .
</item 1>

<item 2>
Vanishing point
With the assumption that the environment consists of objects with only parallel or perpendicular sides , also called Lego - land , using vanishing points
constructed in a single image of the scene they recovered the 3D geometry of the scene . Similar ideas are also used in the field of robotics , mainly in
navigation and autonomous vehicles , and in areas concerned with object detection . See also ( edit ) Graphical projection References ( edit ) ^ Jump up to :
Kirsti Andersen ( 2007 ) Geometry of an Art , p. xxx , Springer , ISBN 0 - 387 - 25961 - 9 ^ Jump up to : B. Caprile , V. Torre ( 1 ) `` Using Vanishing Points
for Camera Calibration '' , International Journal of Computer Vision , Volume 4 , Issue 2 , pp. 127 - 139 , March 1990 Jump up ^ H. Ditton ( 1712 ) Treatise on
Perspective , p 45 Jump up ^ S.T. Barnard ' Interpreting Perspective Images `` , Artificial Intelligence 21 , 1983 , pp.
</item 2>

<item 3>
Vanishing point
This is one - point perspective . Similarly , when the image plane intersects two world - coordinate axes , lines parallel to those planes will meet at infinity
and form two vanishing points . This is called two - point perspective . In three - point perspective the image plane intersects the x , y , and z axes and
therefore lines parallel to these axes intersect , resulting in three different vanishing points . See also : Property 1 below Theorem ( edit ) The vanishing
point theorem is the principal theorem in the science of perspective . It says that the image in a picture plane π of a line L in space , not parallel to the
picture , is determined by its intersection with π and its vanishing point . Some authors have used the phrase , `` the image of a line includes its vanishing
point '' . Guidobaldo del Monte gave several verifications , and Humphry Ditton called the result the `` main and Great Proposition '' . Brook Taylor wrote the
first book in English on perspective in 1714 , which introduced the term `` vanishing point '' and was the first to fully explain the geometry of multipoint
perspective , and historian Kirsti Andersen compiled these observations .
</item 3>

<item 4>
Vanishing point
say D , will have the same vanishing point or converge at the same vanishing points . Mathematically , let q ≡ ( x , y , f ) be a point lying on the image plane
, where f is the focal length ( of the camera associated with the image ) , and let v ≡ ( x / h , y / h , f / h ) be the unit vector associated with q , where h
= √ x + y + f . If we consider a straight line in space S with the unit vector n ≡ ( n , n , n ) and its vanishing point v , the unit vector associated with v
is equal to n , assuming both are assumed to point towards the image plane . When the image plane is parallel to two world - coordinate axes , lines parallel to
the axis which is cut by this image plane will meet at infinity i.e. at the vanishing point . Lines parallel to the other two axes will not form vanishing
points as they are parallel to the image plane . This is one - point perspective .
</item 4>

<item 5>
Vanishing point
If we know the coordinates of one of these points , say v , and the direction of a straight line on the image plane , which passes through a second point , say
v , we can compute the coordinates of both v and v 3 . Let A , B , and C be three mutually orthogonal straight lines in space and v ≡ ( x , y , f ) , v ≡ ( x ,
y , f ) , v ≡ ( x , y , f ) be the three corresponding vanishing points respectively . The orthocenter of the triangle with vertices in the three vanishing
points is the intersection of the optical axis and the image plane . Curvilinear and reverse perspective ( edit ) A curvilinear perspective is a drawing with
either 4 or 5 vanishing points . In 5 - point perspective the vanishing points are mapped into a circle with 4 vanishing points at the cardinal headings N , W ,
S , E and one at the circle 's origin . A reverse perspective is a drawing with vanishing points that are placed outside the painting with the illusion that
they are `` in front of '' the painting . Single point perspective projection . Double point perspective projection .
</item 5>

<item 6>
Drawing
Typically this convergence point is somewhere along the horizon , as buildings are built level with the flat surface . When multiple structures are aligned with
each other , such as buildings along a street , the horizontal tops and bottoms of the structures typically converge at a vanishing point . When both the fronts
and sides of a building are drawn , then the parallel lines forming a side converge at a second point along the horizon ( which may be off the drawing paper . )
This is a two - point perspective . Converging the vertical lines to a third point above or below the horizon then produces a three - point perspective . Depth
can also be portrayed by several techniques in addition to the perspective approach above . Objects of similar size should appear ever smaller the further they
are from the viewer . Thus the back wheel of a cart appears slightly smaller than the front wheel . Depth can be portrayed through the use of texture . As the
texture of an object gets further away it becomes more compressed and busy , taking on an entirely different character than if it was close . Depth can also be
portrayed by reducing the contrast in more distant objects , and by making their colors less saturated .
</item 6>

<item 7>
Vanishing point
The horizon line is a theoretical line that represents the eye level of the observer . If the object is below the horizon line , its vanishing lines angle up to
the horizon line . If the object is above , they slope down . All vanishing lines end at the horizon line . Properties of vanishing points ( edit ) 1 .
Projections of two sets of parallel lines lying in some plane π appear to converge , i.e. the vanishing point associated with that pair , on a horizon line , or
vanishing line H formed by the intersection of the image plane with the plane parallel to π and passing through the pinhole . Proof : Consider the ground plane
π , as y = c which is , for the sake of simplicity , orthogonal to the image plane . Also , consider a line L that lies in the plane π , which is defined by the
equation ax + bz = d .
</item 7>

<item 8>
Vanishing point
Vanishing point - wikipedia Vanishing point Jump to : navigation , search For other uses , see Vanishing point ( disambiguation ) . A vanishing point can be
seen at the far end of this railroad . A vanishing point is a point on the image plane of a perspective drawing where the two - dimensional perspective
projections ( or drawings ) of mutually parallel lines in three - dimensional space appear to converge . When the set of parallel lines is perpendicular to a
picture plane , the construction is known as one - point perspective , and their vanishing point corresponds to the oculus , or `` eye point '' , from which the
image should be viewed for correct perspective geometry . Traditional linear drawings use objects with one to three sets of parallels , defining one to three
vanishing points . Part of a series on Graphical projection Planar ( show ) Parallel projection Orthographic projection Multiview projection Axonometric
projection Isometric projection Oblique projection Perspective projection Curvilinear perspective Reverse perspective Views ( show ) 2.
</item 8>

<item 9>
Vanishing point
5 D Bird 's - eye view Cross section Cutaway drawing Exploded view drawing Fisheye lens Panorama Worm 's - eye view Zoom lens Topics ( show ) 3D projection
Anamorphosis Axonometry Computer graphics Computer - aided design Descriptive geometry Engineering drawing Map projection Picture plane Plans ( drawings )
Projection ( linear algebra ) Projection plane Projective geometry Stereoscopy Technical drawing Vanishing point Video game graphics Viewing frustum Contents (
hide ) 1 Vector notation 2 Theorem 3 Vanishing line 4 Properties of vanishing points 5 Curvilinear and reverse perspective 6 Detection of vanishing points 7
Applications of vanishing points 8 See also 9 References 10 External links Vector notation ( edit ) A 2D construction of perspective viewing , showing the
formation of a vanishing point The vanishing point may also be referred to as the `` direction point '' , as lines having the same directional vector , say D ,
will have the same vanishing point or converge at the same vanishing points .
</item 9>

<item 10>
Vanishing point
She notes , in terms of projective geometry , the vanishing point is the image of the point at infinity associated with L , as the sightline from O through the
vanishing point is parallel to L . Vanishing line ( edit ) As a vanishing point originates in a line , so a vanishing line originates in a plane α that is not
parallel to the picture π . Given the eye point O , and β the plane parallel to α and lying on O , then the vanishing line of α is β ∩ π . For example , when α
is the ground plane and β is the horizon plane , then the vanishing line of α is the horizon line β ∩ π . Anderson notes , `` Only one particular vanishing line
occurs , often referred to as the `` horizon '' . To put it simply , the vanishing line of some plane , say α , is obtained by the intersection of the image
plane with another plane , say β , parallel to the plane of interest ( α ) , passing through the camera center . For different sets of lines parallel to this
plane α , their respective vanishing points will lie on this vanishing line .
</item 10>
=== END ITEM id=bc4ab6e25bbbb224 ===
