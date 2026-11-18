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

=== ITEM id=9e5329bc148cd402 ===
Gold sentences and facts:
Sentence 0: For thousands of years , seawater pearls were retrieved by divers in the Indian Ocean in areas such as the Persian Gulf , the Red Sea and the Gulf of Mannar . Evidence also suggest a prehistoric origin to pearl diving in these regions . Starting in the Han Dynasty ( 206 BC -- 220 AD ) , the Chinese hunted extensively for seawater pearls in the South China Sea . In the 14th - century Arabian Sea , the traveller Ibn Battuta provided the earliest known description of pearl diving by means of attaching a cord to the diver 's waist .
  Fact 0.0: For thousands of years, seawater pearls were retrieved by divers in the Indian Ocean.
  Fact 0.1: Seawater pearls were retrieved by divers in the Persian Gulf.
  Fact 0.2: Seawater pearls were retrieved by divers in the Red Sea.
  Fact 0.3: Seawater pearls were retrieved by divers in the Gulf of Mannar.
  Fact 0.4: Evidence suggests a prehistoric origin to pearl diving in the Persian Gulf, the Red Sea and the Gulf of Mannar.
  Fact 0.5: Starting in the Han Dynasty (206 BC to 220 AD), the Chinese hunted extensively for seawater pearls in the South China Sea.
  Fact 0.6: In the 14th-century Arabian Sea, the traveller Ibn Battuta provided the earliest known description of pearl diving.
  Fact 0.7: Ibn Battuta's description of pearl diving involved attaching a cord to the diver's waist.

Retrieved items:
<item 1>
Graph query row:
entity: pearl diving
water_body: Gulf of Mannar
relationship: LOCATED_IN
</item 1>

<item 2>
Pearl
While white , and more recently black , saltwater pearls are by far the most popular , other color tints can be found on pearls from the oceans . Pink , blue ,
champagne , green , black and even purple saltwater pearls can be encountered , but to collect enough of these rare colors to form a complete string of the same
size and same shade can take years . Religious references ( edit ) This section needs additional citations for verification . Please help improve this article
by adding citations to reliable sources . Unsourced material may be challenged and removed . ( March 2018 ) ( Learn how and when to remove this template message
) Hindu scriptures ( edit ) The Hindu tradition describes the sacred Nine Pearls which were first documented in the Garuda Purana , one of the books of the
Hindu mythology . Ayurveda contains references to pearl powder as a stimulant of digestion and to treat mental ailments . According to Marco Polo , the kings of
Malabar wore a necklace of 104 rubies and pearls which was given from one generation of kings to the next . The reason was that every king had to say 104
prayers every morning and every evening .
</item 2>

<item 3>
Pearl
In 2010 , China overtook Japan in akoya pearl production . Japan has all but ceased its production of akoya pearls smaller than 8 mm . Japan maintains its
status as a pearl processing center , however , and imports the majority of Chinese akoya pearl production . These pearls are then processed ( often simply
matched and sorted ) , relabeled as product of Japan , and exported . In the past two decades , cultured pearls have been produced using larger oysters in the
south Pacific and Indian Ocean . The largest pearl oyster is the Pinctada maxima , which is roughly the size of a dinner plate . South Sea pearls are
characterized by their large size and warm luster . Sizes up to 14 mm in diameter are not uncommon . In 2013 , Indonesia Pearl supplied 43 percent of South Sea
Pearls international market . The other significant producers are Australia , Philippines , Myanmar and Malaysia . Freshwater pearl farming ( edit ) In 1914 ,
pearl farmers began growing cultured freshwater pearls using the pearl mussels native to Lake Biwa . This lake , the largest and most ancient in Japan , lies
near the city of Kyoto .
</item 3>

<item 4>
Pearl
It also records that eight varieties of pearls accompanied Prince Vijaya 's embassy to the Pandyan king as well as king Devanampiya Tissa 's embassy to Emperor
Ashoka . Pliny the Elder ( 23 -- 79AD ) praised the pearl fishery of the Gulf as most productive in the world . For thousands of years , seawater pearls were
retrieved by divers in the Indian Ocean in areas such as the Persian Gulf , the Red Sea and the Gulf of Mannar . Evidence also suggest a prehistoric origin to
pearl diving in these regions . Starting in the Han Dynasty ( 206 BC -- 220 AD ) , the Chinese hunted extensively for seawater pearls in the South China Sea .
In the 14th - century Arabian Sea , the traveller Ibn Battuta provided the earliest known description of pearl diving by means of attaching a cord to the diver
's waist . Catching of pearls , Bern Physiologus ( 9th century ) When Spanish conquistadors arrived in the Western Hemisphere , they discovered that around the
islands of Cubagua and Margarita , some 200 km north of the Venezuelan coast , was an extensive pearl bed ( a bed of pearl oysters ) .
</item 4>

<item 5>
Pearl
Freshwater and saltwater pearls ( edit ) Freshwater and saltwater pearls may sometimes look quite similar , but they come from different sources . Freshwater
pearls form in various species of freshwater mussels , family Unionidae , which live in lakes , rivers , ponds and other bodies of fresh water . These
freshwater pearl mussels occur not only in hotter climates , but also in colder more temperate areas such as Scotland ( where they are protected under law ) .
Most freshwater cultured pearls sold today come from China . Saltwater pearls grow within pearl oysters , family Pteriidae , which live in oceans . Saltwater
pearl oysters are usually cultivated in protected lagoons or volcanic atolls . Creation ( edit ) Diagram comparing a cross-section of a cultured pearl , upper ,
with a natural pearl , lower Pearls are formed inside the shell of certain mollusks as a defense mechanism against a potentially threatening irritant such as a
parasite inside the shell , or an attack from outside that injures the mantle tissue . The mollusk creates a pearl sac to seal off the irritation .
</item 5>

<item 6>
Pearl
Prized for their exquisitely beautiful ' orient ' or lustre , South Sea pearls are now farmed in various parts of the world where the Pinctada maxima oysters
can be found , with the finest South Sea pearls being produced by Paspaley along the remote coastline of North - Western Australia . White and silver colored
South Sea pearls tend to come from the Broome area of Australia , while golden colored ones are more prevalent in the Philippines and Indonesia . A farm in the
Gulf of California , Mexico , is culturing pearls from the black lipped Pinctada mazatlanica oysters and the rainbow lipped Pteria sterna oysters . Also called
Concha Nácar , the pearls from these rainbow lipped oysters fluoresce red under ultraviolet light . From other species ( edit ) A shell of the Indian volute ,
Melo melo , surrounded by a number of pearls from this species Conch pearl pendant Biologically speaking , under the right set of circumstances , almost any
shelled mollusk can produce some kind of pearl . However , most of these molluskan pearls have no luster or iridescence .
</item 6>

<item 7>
Pearl
They are , for the most part , freshwater pearls from mussels . Pearling was banned in the U.K. in 1998 due to the endangered status of river mussels .
Discovery and publicity about the sale for a substantial sum of the Abernethy pearl in the River Tay had resulted in heavy exploitation of mussel colonies
during the 1970s and 80s by weekend warriors . When it was permitted it was carried on mainly by Scottish Travellers who found pearls varied from river to river
with the River Oykel in the Highlands being noted for the finest rose - pink pearls . There are two firms in Scotland that are licensed to sell pre-1998
freshwater pearls . Pearl farming ( edit ) A pearl being extracted from an akoya pearl oyster . See also : Oyster farming Today , the cultured pearls on the
market can be divided into two categories . The first category covers the beaded cultured pearls , including Akoya , South Sea and Tahiti . These pearls are
gonad grown , and usually one pearl is grown at a time . This limits the number of pearls at a harvest period .
</item 7>

<item 8>
Pearl
1 British Isles 7.2 Pearl farming 8 Timeline of pearl production 9 Freshwater pearl farming 10 Momme weight 11 In jewelry 11.1 Gallery 11.2 Shapes 11.3 Lengths
of pearl necklaces 11.4 Colors 12 Religious references 12.1 Hindu scriptures 12.2 Hebrew scriptures 12.3 New Testament scriptures 12.4 Islamic scriptures 12.5
Additional references 13 See also 14 References 15 External links Etymology ( edit ) The English word pearl comes from the French perle , originally from the
Latin perna meaning leg , after the ham - or mutton leg - shaped bivalve . Definition ( edit ) A black pearl and a shell of the black - lipped pearl oyster .
The iridescent colors originate from nacre layers . All shelled mollusks can , by natural processes , produce some kind of `` pearl '' when an irritating
microscopic object becomes trapped within its mantle folds , but the great majority of these `` pearls '' are not valued as gemstones .
</item 8>

<item 9>
Pearl
Journal of the Marine Biological Association of India . 15 : 1 -- 19 . Archived ( PDF ) from the original on July 18 , 2011 . Jump up ^ Arnold Wright ( 1999 ) .
Twentieth century impressions of Ceylon : its history , people , commerce , industries , and resources . p. 227 . ISBN 81 - 206 - 1335 - X . Archived from the
original on January 18 , 2017 . Jump up ^ James Hornell ( 2009 ) . The Indian Pearl Fisheries of the Gulf of Manar and Palk Bay . BiblioBazaar . p. 6 . ISBN 1 -
110 - 87096 - 5 . Archived from the original on January 18 , 2017 . Jump up ^ De Silva , K.M. ( 1995 ) . Volume 2 of History of Ceylon , History of Ceylon :
History of Sri Lanka . Peradeniya : Ceylon University Press . p. 56 . ISBN 955 - 589 - 004 - 8 .
</item 9>

<item 10>
Pearl
Australia also has one of the world 's last remaining fleets of pearl diving ships . Australian pearl divers dive for south sea pearl oysters to be used in the
cultured south sea pearl industry . The catch of pearl oysters is similar to the numbers of oysters taken during the natural pearl days . Hence significant
numbers of natural pearls are still found in the Australian Indian Ocean waters from wild oysters . X-ray examination is required to positively verify natural
pearls found today . Types of cultured pearls ( edit ) A blister pearl , a half - sphere , formed flush against the shell of the pearl oyster . Keshi pearls ,
although they often occur by chance , are not considered natural . They are a byproduct of the culturing process , and hence do not happen without human
intervention . They are quite small , typically only a few millimeters . Keshi pearls are produced by many different types of marine mollusks and freshwater
mussels in China . Keshi pearls are actually a mistake in the cultured pearl seeding process . In seeding the cultured pearl , a piece of mantle muscle from a
sacrificed oyster is placed with a bead of mother of pearl within the oyster .
</item 10>
=== END ITEM id=9e5329bc148cd402 ===

=== ITEM id=feb4561d134a1fe6 ===
Gold sentences and facts:
Sentence 0: When Hoyt Axton performed the song to the group , two of the three main vocalists -- Danny Hutton and Cory Wells -- rejected the song , but Chuck Negron felt that the band needed a `` silly song '' to help bring the band back together as a working unit . Negron also felt that the song `` was n't even close to our best record , but it might have been one of our most honest . ''
  Fact 0.0: Hoyt Axton performed the song to the group.
  Fact 0.1: Two of the three main vocalists, Danny Hutton and Cory Wells, rejected the song.
  Fact 0.2: Chuck Negron felt that the band needed a 'silly song' to help bring the band back together as a working unit.
  Fact 0.3: Chuck Negron also felt that the song was not even close to the band's best record.
  Fact 0.4: Chuck Negron also felt that the song might have been one of the band's most honest.
Sentence 1: The song , which has been described by members of Three Dog Night as a `` kid 's song '' and a `` silly song '' , topped the main singles charts in North America , was certified gold by the RIAA , and has since been covered by multiple artists .
  Fact 1.0: Members of Three Dog Night have described the song as a 'kid's song'.
  Fact 1.1: Members of Three Dog Night have described the song as a 'silly song'.
  Fact 1.2: The song topped the main singles charts in North America.
  Fact 1.3: The song was certified gold by the RIAA.
  Fact 1.4: The song has since been covered by multiple artists.

Retrieved items:
<item 1>
Joy to the World (Three Dog Night song)
Genre Rock Length 3 : 50 ( album ) 3 : 17 ( single ) Label Dunhill Songwriter ( s ) Hoyt Axton Producer ( s ) Richard Podolor Three Dog Night singles chronology
`` One Man Band '' ( 1970 ) `` Joy to the World '' ( 1971 ) `` Liar '' ( 1971 ) `` One Man Band '' ( 1970 ) `` Joy to the World '' ( 1971 ) `` Liar '' ( 1971 )
`` Joy to the World '' is a song written by Hoyt Axton , and made famous by the band Three Dog Night . The song is also popularly known by its opening lyric ,
`` Jeremiah was a bullfrog '' . Three Dog Night originally released the song on their fourth studio album , Naturally in November 1970 and subsequently released
an edited version of the song as a single in February 1971 .
</item 1>

<item 2>
Joy to the World (Three Dog Night song)
Entertainment section . August 20 , 2008 . ISSN 1053 - 2889 . Retrieved April 5 , 2011 . Jump up ^ `` Jackson v. Axton '' . Law.justia.com . Retrieved 2016 - 10
- 04 . Jump up ^ `` Forum - 1970 ( ARIA Charts : Special Occasion Charts ) '' . Australian-charts.com . Archived from the original on 2016 - 10 - 20 . Retrieved
2017 - 05 - 07 . Jump up ^ `` 100 Singles '' ( PHP ) . RPM. 15 ( 15 ) . May 29 , 1971 . Retrieved April 5 , 2011 . Jump up ^ `` Musicline.de -- Three Dog Night
Single - Chartverfolgung '' ( in German ) . Media Control Charts . PhonoNet GmbH . Jump up ^ `` Dutchcharts.nl -- Three Dog Night -- Joy To The World '' ( in
Dutch ) . Single Top 100 .
</item 2>

<item 3>
Joy to the World (Three Dog Night song)
1965 -- 1975 ( 1993 ) 20th Century Masters -- The Millennium Collection : The Best of Three Dog Night ( 1999 ) Singles `` Nobody '' `` Try a Little Tenderness
'' `` One '' `` Easy to Be Hard '' `` Eli 's Coming '' `` Celebrate '' `` Mama Told Me ( Not to Come ) '' `` Out in the Country '' `` One Man Band '' `` Joy to
the World '' `` Liar '' `` An Old Fashioned Love Song '' `` Never Been to Spain '' `` The Family of Man '' `` Black & White '' `` Pieces of April '' `` Shambala
'' `` Let Me Serenade You '' `` The Show Must Go On '' `` Sure As I 'm Sittin ' Here '' `` Play Something Sweet ( Brickyard Blues ) '' `` Til the World Ends ''
`` Everybody Is a Masterpiece '' `` It 's a Jungle Out There '' Discography Book Billboard Year - End number one singles ( 1960 -- 1979 ) 1960 : `` Theme from A
Summer Place '' -- Percy Faith 1961 : `` Tossin ' and Turnin ' '' -- Bobby Lewis 1962 : ``
</item 3>

<item 4>
Joy to the World (Three Dog Night song)
The song , which has been described by members of Three Dog Night as a `` kid 's song '' and a `` silly song '' , topped the main singles charts in North
America , was certified gold by the RIAA , and has since been covered by multiple artists . The song is featured prominently in the film The Big Chill . It is
sung by a child character at the beginning and the Three Dog Night recording is played over the end credits . It is also played at the end of every Denver
Broncos home victory . Notable playings of this song after Broncos victories included then - Chicago Bears head coach Abe Gibron 's singing along with the song
in 1973 ; and at the end of Super Bowl XXXII , played at Qualcomm Stadium in San Diego . It was also played at the end of Super Bowl XXXIII at Pro Player ( now
Hard Rock ) Stadium in Miami Gardens , Florida and Super Bowl 50 in Santa Clara , California . Contents ( hide ) 1 Background and recording 2 Charts and awards
3 In popular culture 4 References 5 External links Background and Recording ( edit ) Some of the words are nonsensical .
</item 4>

<item 5>
Joy to the World (Three Dog Night song)
wikipedia.org/w/index.php?title=Joy_to_the_World_(Three_Dog_Night_song)&oldid=835489072 '' Categories : 1971 singles Billboard Hot 100 number - one singles
Three Dog Night songs Dunhill Records singles Songs written by Hoyt Axton Songs about animals Novelty songs RPM Top Singles number - one singles 1970 songs
Hidden categories : Articles needing additional references from June 2008 All articles needing additional references Articles with hAudio microformats
Singlechart usages for German Singlechart usages for Dutch100 Singlechart usages for UK Singlechart usages for Billboardhot100 Singlechart called without song
Talk Contents About Wikipedia 日本 語 Edit links This page was last edited on 9 April 2018 , at 00 : 34 . Text is available under the Creative Commons Attribution
- ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered
trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 5>

<item 6>
“Joy to the World,” written by Hoyt Axton and recorded by Three Dog Night, first appeared on their 1970 album *Naturally*; an edited single followed in February
1971. Known for “Jeremiah was a bullfrog,” the playful song featured all seven band members singing. It topped the U.S. Billboard Hot 100 for six weeks, was
1971’s year-end No. 1, earned gold certification, and sold about five million copies worldwide.
</item 6>

<item 7>
Joy to the World (Three Dog Night song)
Joy to the World ( Three Dog Night song ) - wikipedia Joy to the World ( Three Dog Night song ) Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
June 2008 ) ( Learn how and when to remove this template message ) `` Joy to the World '' 1971 French release Single by Three Dog Night from the album Naturally
B - side `` I Can Hear You Calling '' Released February 1971 ( 1971 - 02 ) Format 7 '' 45 RPM Recorded 1970 at American Recording Co .
</item 7>

<item 8>
Joy to the World (Three Dog Night song)
Axton wanted to persuade his record producers to record a new melody he had written and the producers asked him to sing any words to the tune . A member of
Three Dog Night said that the original lyrics to the song were `` Jeremiah was a prophet '' but `` no one liked that '' . When Hoyt Axton performed the song to
the group , two of the three main vocalists -- Danny Hutton and Cory Wells -- rejected the song , but Chuck Negron felt that the band needed a `` silly song ''
to help bring the band back together as a working unit . Negron also felt that the song `` was n't even close to our best record , but it might have been one of
our most honest . '' The song was recorded by Three Dog Night at American Recording Company , produced by Richard Podolor , and engineered by Bill Cooper .
Unlike most Three Dog Night songs recorded at that point , instead of having just the three main vocalists singing harmony , the song was recorded with all
seven members of the band singing . Drummer Floyd Sneed sings the deep lyric `` I wan na tell you '' towards the end of the song .
</item 8>

<item 9>
Joy to the World (Three Dog Night song)
ispot.tv . Retrieved December 24 , 2017 . External links ( edit ) Lyrics of this song at MetroLyrics Three Dog Night Danny Hutton Michael Allsup Paul Kingery
Pat Bautz Eddie Reasoner David Morgan Cory Wells Chuck Negron Jimmy Greenspoon Floyd Sneed Joe Schermie Skip Konte Al Ciner Gary Moon Richard Campbell Studio
albums Three Dog Night ( 1968 ) Suitable for Framing ( 1969 ) It Ai n't Easy ( 1970 ) Naturally ( 1970 ) Harmony ( 1971 ) Seven Separate Fools ( 1972 ) Cyan (
1973 ) Hard Labor ( 1974 ) Coming Down Your Way ( 1975 ) American Pastime ( 1976 ) It 's a Jungle ( 1983 ) Live albums Captured Live at the Forum ( 1969 )
Around the World with Three Dog Night ( 1973 ) Compilations Golden Bisquits ( 1971 ) Joy to the World : Their Greatest Hits ( 1975 ) The Best of 3 Dog Night (
1982 ) Celebrate : The Three Dog Night Story , 1965 -- 1975 (
</item 9>

<item 10>
Joy to the World (Three Dog Night song)
The single went on to sell 5 million copies worldwide , making it one of the best - selling singles of all time . In popular culture ( edit ) Singer -
songwriter Daniel Johnston likes this song , which inspired his own character Jeremiah the bullfrog , whom he often draws in his artwork and advertisements .
The frog is also featured on the cover of his album Hi , How Are You ( 1983 ) and has become an official music mascot for Johnston . It is sung by the son of
one of the main characters at the start of the film The Big Chill ( 1983 ) and is featured on the soundtrack . In The X-Files TV series ( Detour - S05E04 ) ,
Scully sings this song while she 's lost in a forest with Mulder to show him she is awake and he can rest . It is played during the end credits of the R - rated
animated film , Sausage Party ( 2016 ) , and later in the end credits of the PG - 13 - rated animated film A Very Spooky Night . It appears in 2017 and 2018 TV
commercials for Big Lots .
</item 10>
=== END ITEM id=feb4561d134a1fe6 ===

=== ITEM id=689fe10b0a0502d5 ===
Gold sentences and facts:
Sentence 0: No . Official English title Original Japanese title Original air date English air date 151 `` Sabertooth '' `` Seibātūsu '' ( 剣 咬 の 虎 ( セイバー トゥース ) ) October 6 , 2012 November 17 , 2015 The Legion Platoon is tasked by Zentopia 's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again . Before embarking on their mission , they visit Fairy Tail and make amends with the guild for their previous hostility against them . It is later discovered that Fairy Tail has lost their title as the strongest guild in Fiore to the Saber Tooth guild over the past seven years . Two of the strongest wizards in the guild include a pair of Dragon Slayers named Sting and Rogue , who are obsessed with fighting Natsu and Gajeel , respectively . Meanwhile , Makarov brings Gildarts to their abandoned guildhall and shows him Lumen Histoire , a light that Makarov says is Fairy Tail 's greatest secret . Makarov then tells Gildarts that he has been chosen to become the next master of Fairy Tail . 152 `` And So We Aim for the Top '' `` Soshite Ore - tachi wa Chōjō o Mezasu '' ( そして オレ たち は 顶 上 を 目指す ) October 13 , 2012 November 24 , 2015 Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island . The group meets with Poluchka for medicine to strengthen their magic , but are unsuccessful . Wendy recognizes Poluchka 's voice and scent as those of Grandine , the dragon who raised her . Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her . Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 `` Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction . When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years . They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild , Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power . The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave . 155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray , Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus . As they tour the city , Natsu and Happy encounter the twin dragon slayers , Sting and Rogue , who mock Natsu for not being able to kill Acnologia , and boast of becoming true dragon slayers by killing the dragons who raised them . At midnight , Fairy Tail and the other participating guilds are put through a preliminary event -- a race through a massive floating labyrinth -- designed to narrow the number of participating guilds down to eight teams . 156 `` Sky Labyrinth '' `` Sukai Rabirinsu '' ( 空中 迷宮 ( スカイ ラビリンス ) ) November 10 , 2012 December 22 , 2015 Natsu 's team notices that Wendy has gone missing , so Elfman replaces her to keep them from being eliminated . As they make their way through the enchanted labyrinth , the five encounter various other guilds and steal the maps made by the other guilds to better navigate the maze . They manage to reach the end of the maze and pass the elimination round , confident they have arrived first , only to discover they have barely taken last place . Meanwhile , Happy , Lisanna , and the rest of the guild search for Wendy and Carla . They eventually find the two in the garden of the king 's palace , Mercurius , with their magic power depleted . 157 `` New Guild '' `` Shinki Girudo '' ( 新 规 ギルド ) November 17 , 2012 December 29 , 2015 Wendy 's body is weakened as a result of losing her magic power . As Poluchka helps her recover , she formally requests Elfman to take her place in the Grand Magic Games . Fairy Tail steps into the arena with a negative audience reception , but they are bolstered by their guild mates and the spirit of Mavis , whom only the Fairy Tail guild members can see . The next four participating teams are revealed to be Quatro Cerberus , Mermaid Heel , Blue Pegasus , and Lamia Scale . Following them is Raven Tail , a former dark guild led by Makarov 's son Ivan , who confess to attacking Wendy and Carla . To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced , they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars '' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once . Humiliated , Gray swears revenge against Raven Tail and Saber Tooth . Afterwards , Lucy is chosen to fight Raven Tail member Flare in the tournament 's first battle round . 159 `` Lucy vs. Flare '' `` Rūshii vs. Furea '' ( ルーシィ vs. フレア ) December 1 , 2012 January 12 , 2016 Lucy displays tremendous skill in her battle against Flare by being able to summon two spirits simultaneously . As Lucy gains the upper hand , Flare cheats by using her magic hair to hold Alzack and Bisca 's daughter Asuka as a hostage , blackmailing Lucy into not fighting back to keep her from harming Asuka . As Lucy tearfully speaks of Asuka 's danger , Natsu is able to hear it and rushes to Asuka to incinerate Flare 's hair , allowing Lucy to counterattack with her Urano Metoria spell in combination with Gemini . However , before the spell fully activates , Flare 's teammate Obra nullifies Lucy 's magic , causing her to collapse and lose the match . 160 `` Portent '' `` Kyōzui '' ( 凶 瑞 ) December 8 , 2012 January 19 , 2016 Blue Pegasus ' Ren defeats Mermaid Heel 's Araña during the games ' second match , while Saber Tooth 's Orga effortlessly defeats Quatro Cerberus ' War Cry in the third match . For the day 's fourth and final match , Jellal is pitted against Lamia Scale 's Jura , intending to avenge the humiliated Fairy Tail members . He mimics Mystogan 's magic to keep up his disguise , but he finds it ineffective to use against Jura and is forced to use his own magic instead . Before he can cast a spell that will expose his true identity , Ultear uses Merudy 's sensory to force Jellal into submission . Meanwhile , the hospitalized Carla awakens after having a troubling premonition of the future . 161 `` Chariots '' `` Chariotto '' ( 战 车 ( チャリオット ) ) December 15 , 2012 January 26 , 2016 Fairy Tail is undaunted by their poor performance on the first day . As the guild celebrates , they are visited and taunted by Bacchus , Erza 's rival from Quatro Cerberus , who has come to replace War Cry . Carla relays her premonition to Poluchka , saying that she saw Mercurius collapsing around what appeared to be a tearfully singing Lucy . The second day of the tournament begins with a footrace atop a train of moving chariots , causing the participants Natsu , Gajeel , and Sting to suffer from motion sickness as a side effect of their Dragon Slayer magic . As Natsu and Gajeel force themselves forward , Sting inquires about their determination to which Natsu responds that it is as a result of the strength of Fairy Tail 's bonds . Sting eventually throws the race while Natsu and Gajeel manage to finish as sixth and seventh placers , respectively , earning Fairy Tail the audience 's respect . 162 `` Elfman vs. Bacchus '' `` Erufuman vs. Bakkasu '' ( エルフマン vs. バッカス ) December 22 , 2012 September 1 , 2015 Due to a miscomunnication between the king and his knight Arcadios , Elfman is chosen as Bacchus ' opponent instead of Erza . Before the match commences , the two opponents make a wager : if Bacchus wins , he will court Elfman 's sisters , and if Elfman wins , Bacchus ' team name will be changed to `` Quatro Puppy '' for the rest of the games . Bacchus ruthlessly attacks Elfman with his bare palms , who manages to damage Bacchus with his defensive Lizardman transformation . Elfman manages to outlast Bacchus as he exhausts himself , winning the match and forcing Bacchus ' team to be renamed . Meanwhile , Natsu finds Wendy , Carla , and Poluchka being kidnapped by a group of masked men . Upon being caught , the men claim they were hired to kidnap Lucy by Raven Tail . They are soon revealed to have actually been sent by Arcadios , who intends to sow discord between the guilds as he plots to use Lucy for his `` Eclipse Project '' . 163 `` Mirajane vs. Jenny '' `` Mirajēn vs. Jenī '' ( ミラ ジェーン vs. ジェニー ) January 5 , 2013 September 1 , 2015 The second day of the Grand Magic Games continues , with Mirajane of Fairy Tail Team B fighting against Jenny Realight of Team Blue Pegasus . Since the two competitors are well - known for being magazine models , their battle consists almost entirely of a fashion showdown , much to the pleasure of the audience . Eventually as the showdown becomes redundant , Jenny proposes a wager : the loser must pose nude for the Sorcerer Magazine , which Mirajane accepts . Jenny then transforms into her battle armor while Mirajane shifts into her Satan Soul `` Sitri '' form and easily defeats Jenny . The final match of the day is announced , with Kagura Mikazuchi of Mermaid Heel pitted against Yukino Aguria of Saber Tooth . Yukino proposes to Kagura that they wager their lives on their battle . Meanwhile , Arcadios is confronted by Darton , the kingdom 's defense minister , about his `` Eclipse Project '' . 164 `` Kagura vs. Yukino '' `` Kagura vs. Yukino '' ( カグラ vs. ユキノ ) January 12 , 2013 September 1 , 2015 Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura . Using her gravity - altering magic , Kagura defeats the two spirits , thus forcing Yukino to use a Gatekey that summons Ophiuchus , which is regarded as the strongest among the Celestial Spirits . However , Kagura slays Ophiuchus with her sword still sheathed and easily defeats Yukino . Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance . Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete '' ( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ; however , Lucy turns down the offer . Meanwhile , Milliana tells Erza that Kagura 's sword , Archenemy , is intended to kill Jellal , and that she and Kagura share the same resentment towards Jellal . Natsu and Happy run after Yukino in an attempt to apologize ; however , Natsu realizes that Yukino had been expelled , and infiltrates the Crocus Garden to fight Jiemma . 166 `` Pandemonium '' `` Pandemoniumu '' ( 伏魔殿 ( パンデモニウム ) ) January 26 , 2013 September 1 , 2015 After Natsu punches out Dobengal , the guild member summoned to fight him , he and Jiemma engage in a duel . However , the guild master 's daughter Minerva appears , ends their duel and , taking Happy hostage , convinces Natsu to settle their dispute in the Grand Magic Games . As Natsu leaves , he states that Saber Tooth will never surpass Fairy Tail because they care for their friends , words that are echoed by Rogue , while Sting is left in awe at Natsu 's strength . The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points . Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent scores depending on the extent of power used . In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei '' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve . For the third match , Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail . Since the opponent is from a suspicious guild , the Fairy Tail members take extra precautions by watching the actions of Ivan Dreyar as well as the other four Raven Tail members . However , as the match commences , the spectators are shocked to see Laxus being beaten by Alexei , although no cautious movements can be seen from the Raven Tail team . It turns out that the two dueling people are just illusions created by Alexei in order for him to have a talk with Laxus . Also , the other Raven Tail members being monitored by Fairy Tail are just lifelike illusions , for in reality they are with Alexei in the battlefield . Alexei reveals himself to be Ivan Dreyar . Unable to interrogate Laxus of the Lumen Histoire 's location , Ivan and the Raven Tail members are forced to fight him . Before their battle begins , Ivan declares that Raven Tail is an anti-Fairy Tail guild . 169 `` Wendy vs. Shelia '' `` Wendi vs. Sheria '' ( ウェンディ vs. シェリア ) February 16 , 2013 September 1 , 2015 After Laxus defeats Ivan and the four Raven Tail elite members , Ivan 's illusions disappear to the shock of everyone in the stadium . The game officials declare Laxus the victor , and Ivan tells him that Lumen Histoire is Fairy Tail 's ' darkness ' . Due to the infraction of the rules , the Raven Tail team is disqualified for the remainder of the Games , and its members are captured by the soldiers . Later , Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off to compete for the final match of the day . As the two exchange attacks , gusts of wind surround the arena since Sherria is revealed to be a Sky God Slayer . Wendy employs an advanced Sky Dragon Slayer attack much to Poluchka 's astonishment , causing Sherria to fall on the ground . Wendy is about to be announced as the winner of the bout when Sherria suddenly stands up with her wounds fully healed . Meanwhile , the Crime Sorcière guild senses the nefarious aura , and Jellal proceeds to the Domus Flau audience , oblivious that he bumped into Doranbolt , who , in turn , wonders what is Mystogan doing in Earthland . 170 `` Small Fists '' `` Chiisana Kobushi '' ( 小さな 拳 ) February 23 , 2013 September 1 , 2015 With Cheria being fully healed and Wendy being almost depleted of magic power , their battle continues , with the former casting a high - level spell towards the latter . However , it seems that Wendy enhances Cheria 's physical strength , causing the attack to gain much force and thus missing Wendy . The two competitors then engage in a hand - to - hand combat , but due to the time limit and neither side is giving up , their battle resulted in a draw , giving both teams with 5 points . Meanwhile , Jellal manages to track the sinister aura among the crowd in the form of a mysterious , hooded person . Jellal is about to confront the aura source , but he is abruptly cornered by Doranbolt , Lahar , and other Rune Knights . Doranbalt then interrogates Jellal , and as his cover was blown , the Rune Knights , as well as Kagura and Milliana glancing nearby , are startled to see the true identity of Mystogan . Yajima convinces the Rune Knights that Mystogan is Jellal 's Edolas counterpart , prompting Lahar to release the fugitive . Meanwhile , Laxus and Makarov have a small conversation regarding Lumen Histoire when Mavis appears and explains that Lumen Histoire is the ' light ' of Fairy Tail . 171 `` Naval Battle '' `` Nabaru Batoru '' ( 海战 ( ナバルバトル ) ) March 2 , 2013 September 1 , 2015 As the jubilant Fairy Tail members celebrate their victories , the other people , such as Sting , Kagura , and Milliana , are contemplating on their own concerns , while Yukino is enlisted as a sergeant under Arcadios ' command . Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others rush to her aid . 172 `` A Parfum for You '' `` Kimi ni Sasageru Parufamu '' ( 君 に 捧げる 香り ( パルファム ) ) March 9 , 2013 September 1 , 2015 While Lucy is hospitalized and taken to the infirmary , the Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so . Later , Makarov announces that the two Fairy Tail teams must merge as a result of the disqualification of Raven Tail and to compensate for the odd number of teams for the battle round . He also reveals that the new team will have the former Team A 's 35 points to start with . The new Fairy Tail team , consisting of Natsu , Gray , Erza , Gajeel , and Laxus , are well - received as they enter the field . The second part of the tournament begins with a tag battle between Blue Pegasus ' Ichiya and `` Rabbit '' and Quatro Puppy 's Rocker and Bacchus . To everyone 's shock , it is revealed that `` Rabbit '' is actually Nichiya in disguise , and after the Exceed is punched by Bacchus , Ichiya bulks up to defeat the Quatro Puppy opponents , awarding Blue Pegasus with 10 points . 173 `` Battle of Dragon Slayers ! '' `` Batoru obu Doragon Sureiyā '' ( バトル ・ オブ ・ ドラゴン スレイヤー ) March 16 , 2013 September 1 , 2015 Mermaid Heel 's Kagura and Milliana and Lamia Scale 's Lyon and Yuka face off for the second tag battle of the games . The two teams struggle to earn a victory , only to have their battle ending in a draw due to the time limit . For the third and final match of the day , the spectators look on in silence as Natsu and Gajeel of Team Fairy Tail and Sting and Rogue of Team Saber Tooth enter the field . As the battle commences , Sting and Rogue were about to take the initiative when they were suddenly attacked by Natsu and Gajeel . The four mages then exchange attacks ; however , the audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers . Unable to keep themselves on par with their adversaries , Sting activates ' White Drive ' while Rogue powers up using ' Shadow Drive ' . 174 `` Four Dragons '' `` Yonin no Doragon '' ( 四 人 の 竜 ( ドラゴン ) ) March 23 , 2013 September 1 , 2015 The tag battle continues , with Sting and Rogue beginning to attack Natsu and Gajeel with their powers being increased . After receiving heavy blows from the Saber Tooth Dragon Slayers , Natsu and Gajeel manages to hold them off , thus forcing Sting and Rogue to activate `` Dragon Force '' on their own will , to the shock of the Fairy Tail audience . Sting decides to finish the Fairy Tail Dragon Slayers alone , for he held a promise to his Exceed friend Lector many years ago . As the three mages fight , the tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel , resulting in the collapse of the arena floor , and after a hand - to - hand combat underground , Sting was seemingly the victor of the match . However , the Fairy Tail duo stands up , comments on Sting 's strength , and argues about his position while attacking , thereby causing Natsu to drag Gajeel on a mining cart and send it away . While Gajeel is left incapacitated due to motion sickness , Natsu challenges the Twin Dragons of Saber Tooth alone , in retribution to what Sting did earlier on them . 175 `` Natsu vs. the Twin Dragons '' `` Natsu vs. Sōryū '' ( ナツ vs. 双竜 ) March 30 , 2013 September 1 , 2015 The tag battle of the Dragon Slayers comes to a close , with Natsu overpowering the combined attacks of Sting and Rogue . The Saber Tooth members watch in dismay , while Jiemma seethes in anger over the defeat of the Twin Dragons , frightening the nearby spectators away . The rest of Fairy Tail celebrates in joy as its team receives 10 points , finally surpassing Team Saber Tooth by a single point . Meanwhile , Gajeel , back to his senses after being dragged by Natsu into a cart earlier , stumbles upon a cave full of dragon skeletons . During that night , Jellal manages to corner the source of the dark aura , standing in disbelief at the identity of the mysterious , hooded person .
  Fact 0.0: Episode 151, 'Sabertooth', has the Japanese title 'Seibatusu' and originally aired on October 6, 2012.
  Fact 0.1: Episode 151 aired in English on November 17, 2015.
  Fact 0.2: The Legion Platoon is tasked by Zentopia's archbishop to retrieve the missing pieces of the Infinity Clock and seal them again.
  Fact 0.3: Before embarking on their mission, the Legion Platoon visits Fairy Tail.
  Fact 0.4: The Legion Platoon makes amends with Fairy Tail for their previous hostility against them.
  Fact 0.5: It is later discovered that Fairy Tail has lost its title as the strongest guild in Fiore to Saber Tooth over the past seven years.
  Fact 0.6: Sting and Rogue are two of the strongest wizards in Fairy Tail.
  Fact 0.7: Sting and Rogue are a pair of Dragon Slayers.
  Fact 0.8: Sting is obsessed with fighting Natsu.
  Fact 0.9: Rogue is obsessed with fighting Gajeel.
  Fact 0.10: Makarov brings Gildarts to their abandoned guildhall.
  Fact 0.11: Makarov shows Gildarts Lumen Histoire, a light that Makarov says is Fairy Tail's greatest secret.
  Fact 0.12: Makarov tells Gildarts that he has been chosen to become the next master of Fairy Tail.
  Fact 0.13: Episode 152, 'And So We Aim for the Top', has the Japanese title 'Soshite Ore-tachi wa Chojo o Mezasu' and originally aired on October 13, 2012.
  Fact 0.14: Episode 152 aired in English on November 24, 2015.
  Fact 0.15: Natsu and his friends discover that a power gap has developed between them and the rest of the guild members who were not on Tenrou Island.
  Fact 0.16: The group meets with Poluchka for medicine to strengthen their magic.
  Fact 0.17: The group is unsuccessful in strengthening their magic with the medicine.
  Fact 0.18: Wendy recognizes Poluchka's voice and scent as those of Grandine, the dragon who raised her.
  Fact 0.19: Poluchka reveals herself as the Edolas counterpart of Grandine.
  Fact 0.20: Poluchka gives Wendy written instructions for two advanced spells that Grandine failed to teach her.
  Fact 0.21: Makarov announces Gildarts' succession as master to the rest of the guild.
  Fact 0.22: Makarov finds that Gildarts has gone on a journey, leaving a letter of resignation behind.
  Fact 0.23: In the letter, Gildarts reinstates Laxus into the guild.
  Fact 0.24: In the letter, Gildarts returns mastership to Makarov.
  Fact 0.25: In the letter, Gildarts tells Makarov to make Fairy Tail the strongest guild in Fiore again.
  Fact 0.26: Romeo suggests they participate in the Grand Magic Games.
  Fact 0.27: The Grand Magic Games are an annual festival designed to determine the guild rankings in Fiore.
  Fact 0.28: Episode 153, 'Song of the Stars', has the Japanese title 'Hoshiboshi no Uta' and originally aired on October 20, 2012.
  Fact 0.29: Episode 153 aired in English on December 1, 2015.
  Fact 0.30: The Grand Magic Games are occurring in three months' time.
  Fact 0.31: Natsu and his friends train at a beach resort.
  Fact 0.32: On their second day of training, Virgo appears and tells the wizards that the Celestial Spirit World is in danger of destruction.
  Fact 0.33: When the wizards arrive, they find a surprise party held by the Celestial Spirit King.
  Fact 0.34: The party celebrates the wizards' return from Sirius Island after seven years.
  Fact 0.35: The wizards spend the day in the spirit world partying with the spirits.
  Fact 0.36: Virgo informs the wizards that one day in the spirit world equals three months in the human world.
  Fact 0.37: The wizards find that the Grand Magic Games are upon them.
  Fact 0.38: Episode 154, 'For All the Time We Missed Each Other', has the Japanese title 'Surechigatta Jikan no Bun dake' and originally aired on October 27, 2012.
  Fact 0.39: Episode 154 aired in English on December 8, 2015.
  Fact 0.40: While mulling over their lost training time, the Fairy Tail wizards receive a message via carrier pigeon.
  Fact 0.41: The message requests a private meeting in the forest nearby.
  Fact 0.42: At the meeting, the wizards meet a runaway Jellal, Ultear, and Merudy.
  Fact 0.43: Jellal, Ultear, and Merudy have formed their own guild, Crime Sorciere.
  Fact 0.44: Crime Sorciere aims to defeat Zeref and any dark guilds that try to use his power.
  Fact 0.45: Jellal, Ultear, and Merudy ask Fairy Tail for help in determining the source of a magical anomaly similar to Zeref's magic.
  Fact 0.46: The magical anomaly has been sensed around the Grand Magic Games arena, Domus Flau.
  Fact 0.47: In exchange, Ultear uses her magic to invoke Fairy Tail's Second Origin in order to double their magic power.
  Fact 0.48: Erza talks with Jellal over his regained memories.
  Fact 0.49: Erza and Jellal nearly kiss.
  Fact 0.50: Jellal decides that he and Erza can not be together.
  Fact 0.51: Jellal lies to Erza that he has a fiancee.
  Fact 0.52: Erza quickly picks up on the lie and respects his decision.
  Fact 0.53: Erza sees Crime Sorciere off as they leave.
  Fact 0.54: Episode 155, 'Crocus, the Flower-Blooming Capital', has the Japanese title 'Hanasaku Miyako: Kurokkasu' and originally aired on November 3, 2012.
  Fact 0.55: Episode 155 aired in English on December 15, 2015.
  Fact 0.56: Natsu, Lucy, Gray, Erza, and Wendy are chosen to represent their guild in the Grand Magic Games.
  Fact 0.57: The Grand Magic Games are to be held in Fiore's capital, Crocus.
  Fact 0.58: As they tour the city, Natsu and Happy encounter the twin dragon slayers, Sting and Rogue.
  Fact 0.59: Sting and Rogue mock Natsu for not being able to kill Acnologia.
  Fact 0.60: Sting and Rogue boast of becoming true dragon slayers by killing the dragons who raised them.
  Fact 0.61: At midnight, Fairy Tail and the other participating guilds are put through a preliminary event.
  Fact 0.62: The preliminary event is a race through a massive floating labyrinth.
  Fact 0.63: The preliminary event is designed to narrow the number of participating guilds down to eight teams.
  Fact 0.64: Episode 156, 'Sky Labyrinth', has the Japanese title 'Sukai Rabirinsu' and originally aired on November 10, 2012.
  Fact 0.65: Episode 156 aired in English on December 22, 2015.
  Fact 0.66: Natsu's team notices that Wendy has gone missing.
  Fact 0.67: Elfman replaces Wendy to keep Natsu's team from being eliminated.
  Fact 0.68: As they make their way through the enchanted labyrinth, the five encounter various other guilds.
  Fact 0.69: The five steal the maps made by the other guilds to better navigate the maze.
  Fact 0.70: The five reach the end of the maze and pass the elimination round.
  Fact 0.71: The five are confident they have arrived first.
  Fact 0.72: The five discover they have barely taken last place.
  Fact 0.73: Happy, Lisanna, and the rest of the guild search for Wendy and Carla.
  Fact 0.74: Happy, Lisanna, and the rest of the guild find Wendy and Carla in the garden of the king's palace, Mercurius.
  Fact 0.75: Wendy and Carla have their magic power depleted.
  Fact 0.76: Episode 157, 'New Guild', has the Japanese title 'Shinki Girudo' and originally aired on November 17, 2012.
  Fact 0.77: Episode 157 aired in English on December 29, 2015.
  Fact 0.78: Wendy's body is weakened as a result of losing her magic power.
  Fact 0.79: Poluchka helps Wendy recover.
  Fact 0.80: Wendy formally requests Elfman to take her place in the Grand Magic Games.
  Fact 0.81: Fairy Tail steps into the arena with a negative audience reception.
  Fact 0.82: Fairy Tail is bolstered by their guild mates and the spirit of Mavis, whom only Fairy Tail guild members can see.
  Fact 0.83: The next four participating teams are Quatro Cerberus, Mermaid Heel, Blue Pegasus, and Lamia Scale.
  Fact 0.84: Raven Tail is a former dark guild led by Makarov's son Ivan.
  Fact 0.85: Raven Tail members confess to attacking Wendy and Carla.
  Fact 0.86: The next team is a second team from Fairy Tail composed of Gajeel, Juvia, Mirajane, Laxus, and Jellal.
  Fact 0.87: Jellal is disguised as Mystogan to help investigate the arena.
  Fact 0.88: The final guild to be announced is Saber Tooth.
  Fact 0.89: The audience cheers joyfully for Saber Tooth's announcement.
  Fact 0.90: All eight teams are announced.
  Fact 0.91: Each team sends out one member to participate in the first contest of the games.
  Fact 0.92: Gray and Juvia represent their teams in the first contest.
  Fact 0.93: Episode 158, 'Night of Shooting Stars', has the Japanese title 'Hoshi Furu Yoru ni' and originally aired on November 24, 2012.
  Fact 0.94: Episode 158 aired in English on January 5, 2016.
  Fact 0.95: The games' first contest, 'Hidden', requires the eight participants to score points by finding and attacking each other.
  Fact 0.96: The contest takes place in a virtual city filled with copies of the participants.
  Fact 0.97: Raven Tail member Nalpudding repeatedly targets Gray.
  Fact 0.98: Gray scores last place in the contest.
  Fact 0.99: Saber Tooth's Rufus scores first by using his Memory-Make magic to attack all the other participants at once.
  Fact 0.100: Gray is humiliated and swears revenge against Raven Tail and Saber Tooth.
  Fact 0.101: Lucy is chosen to fight Raven Tail member Flare in the tournament's first battle round.
  Fact 0.102: Episode 159, 'Lucy vs. Flare', has the Japanese title 'Rushii vs. Furea' and originally aired on December 1, 2012.
  Fact 0.103: Episode 159 aired in English on January 12, 2016.
  Fact 0.104: Lucy displays tremendous skill in her battle against Flare by summoning two spirits simultaneously.
  Fact 0.105: Flare cheats by using her magic hair to hold Alzack and Bisca's daughter, Asuka, as a hostage.
  Fact 0.106: Flare blackmails Lucy into not fighting back to keep Flare from harming Asuka.
  Fact 0.107: Lucy tearfully speaks of Asuka's danger.
  Fact 0.108: Natsu hears Lucy and rushes to Asuka.
  Fact 0.109: Natsu incinerates Flare's hair.
  Fact 0.110: Lucy counterattacks with her Urano Metoria spell in combination with Gemini.
  Fact 0.111: Flare's teammate Obra nullifies Lucy's magic before the spell fully activates.
  Fact 0.112: Lucy collapses and loses the match.
  Fact 0.113: Episode 160, 'Portent', has the Japanese title 'Kyozui' and originally aired on December 8, 2012.
  Fact 0.114: Episode 160 aired in English on January 19, 2016.
  Fact 0.115: Blue Pegasus' Ren defeats Mermaid Heel's Arana during the games' second match.
  Fact 0.116: Saber Tooth's Orga effortlessly defeats Quatro Cerberus' War Cry in the third match.
  Fact 0.117: For the day's fourth and final match, Jellal is pitted against Lamia Scale's Jura.
  Fact 0.118: Jellal intends to avenge the humiliated Fairy Tail members.
  Fact 0.119: Jellal mimics Mystogan's magic to keep up his disguise.
  Fact 0.120: Jellal finds Mystogan's magic ineffective against Jura.
  Fact 0.121: Jellal is forced to use his own magic instead.
  Fact 0.122: Before Jellal can cast a spell that will expose his true identity, Ultear uses Merudy's sensory to force Jellal into submission.
  Fact 0.123: The hospitalized Carla awakens after having a troubling premonition of the future.
  Fact 0.124: Episode 161, 'Chariots', has the Japanese title 'Chariotto' and originally aired on December 15, 2012.
  Fact 0.125: Episode 161 aired in English on January 26, 2016.
  Fact 0.126: Fairy Tail is undaunted by their poor performance on the first day.
  Fact 0.127: As the guild celebrates, they are visited and taunted by Bacchus.
  Fact 0.128: Bacchus is Erza's rival from Quatro Cerberus.
  Fact 0.129: Bacchus has come to replace War Cry.
  Fact 0.130: Carla relays her premonition to Poluchka.
  Fact 0.131: In Carla's premonition, Mercurius is collapsing around what appeared to be a tearfully singing Lucy.
  Fact 0.132: The second day of the tournament begins with a footrace atop a train of moving chariots.
  Fact 0.133: Natsu, Gajeel, and Sting suffer from motion sickness as a side effect of their Dragon Slayer magic.
  Fact 0.134: Natsu responds to Sting that his determination is a result of the strength of Fairy Tail's bonds.
  Fact 0.135: Sting eventually throws the race.
  Fact 0.136: Natsu finishes as sixth place.
  Fact 0.137: Gajeel finishes as seventh place.
  Fact 0.138: Natsu and Gajeel earn Fairy Tail the audience's respect.
  Fact 0.139: Episode 162, 'Elfman vs. Bacchus', has the Japanese title 'Erufuman vs. Bakkasu' and originally aired on December 22, 2012.
  Fact 0.140: Episode 162 aired in English on September 1, 2015.
  Fact 0.141: Due to a miscommunication between the king and his knight Arcadios, Elfman is chosen as Bacchus' opponent instead of Erza.
  Fact 0.142: Before the match, Bacchus and Elfman make a wager.
  Fact 0.143: If Bacchus wins, he will court Elfman's sisters.
  Fact 0.144: If Elfman wins, Bacchus' team name will be changed to 'Quatro Puppy' for the rest of the games.
  Fact 0.145: Bacchus ruthlessly attacks Elfman with his bare palms.
  Fact 0.146: Elfman damages Bacchus with his defensive Lizardman transformation.
  Fact 0.147: Elfman outlasts Bacchus as Bacchus exhausts himself.
  Fact 0.148: Elfman wins the match.
  Fact 0.149: Bacchus' team is renamed.
  Fact 0.150: Natsu finds Wendy, Carla, and Poluchka being kidnapped by a group of masked men.
  Fact 0.151: The masked men claim they were hired to kidnap Lucy by Raven Tail.
  Fact 0.152: The masked men are revealed to have actually been sent by Arcadios.
  Fact 0.153: Arcadios intends to sow discord between the guilds.
  Fact 0.154: Arcadios plots to use Lucy for his 'Eclipse Project'.
  Fact 0.155: Episode 163, 'Mirajane vs. Jenny', has the Japanese title 'Mirajen vs. Jeni' and originally aired on January 5, 2013.
  Fact 0.156: Episode 163 aired in English on September 1, 2015.
  Fact 0.157: The second day of the Grand Magic Games continues.
  Fact 0.158: Mirajane of Fairy Tail Team B fights against Jenny Realight of Team Blue Pegasus.
  Fact 0.159: Mirajane and Jenny are well-known for being magazine models.
  Fact 0.160: Their battle consists almost entirely of a fashion showdown.
  Fact 0.161: Jenny proposes a wager that the loser must pose nude for the Sorcerer Magazine.
  Fact 0.162: Mirajane accepts the wager.
  Fact 0.163: Jenny transforms into her battle armor.
  Fact 0.164: Mirajane shifts into her Satan Soul 'Sitri' form.
  Fact 0.165: Mirajane easily defeats Jenny.
  Fact 0.166: The final match of the day is Kagura Mikazuchi of Mermaid Heel against Yukino Aguria of Saber Tooth.
  Fact 0.167: Yukino proposes to Kagura that they wager their lives on their battle.
  Fact 0.168: Arcadios is confronted by Darton, the kingdom's defense minister, about his 'Eclipse Project'.
  Fact 0.169: Episode 164, 'Kagura vs. Yukino', has the Japanese title 'Kagura vs. Yukino' and originally aired on January 12, 2013.
  Fact 0.170: Episode 164 aired in English on September 1, 2015.
  Fact 0.171: Yukino is revealed to be a celestial spirit wizard when she summons Libra and Pisces to attack Kagura.
  Fact 0.172: Kagura defeats Libra and Pisces using her gravity-altering magic.
  Fact 0.173: Yukino uses a Gatekey that summons Ophiuchus, which is regarded as the strongest among the Celestial Spirits.
  Fact 0.174: Kagura slays Ophiuchus with her sword still sheathed.
  Fact 0.175: Kagura easily defeats Yukino.
  Fact 0.176: Jellal discusses the fact that he has not sensed the evil power similar to Zeref's.
  Fact 0.177: Jellal's discussion takes place after the second day of the Grand Magic Games has ended.
  Fact 0.178: Erza reunites with Milliana.
  Fact 0.179: Milliana is revealed to be a member of Mermaid Heel.
  Fact 0.180: Saber Tooth's master Jiemma is dissatisfied with his guild's performance.
  Fact 0.181: Sting is given another chance.
  Fact 0.182: Jiemma humiliates Yukino for her failure.
  Fact 0.183: Episode 165, 'Hatred at Nightfall', has the Japanese title 'Urami wa Yoru no Tobari ni Tsutsumarete' and originally aired on January 19, 2013.
  Fact 0.184: Episode 165 aired in English on September 1, 2015.
  Fact 0.185: After Yukino is expelled from Saber Tooth, she offers her two golden Gatekeys to Lucy.
  Fact 0.186: Yukino offers the Gatekeys to complete Lucy's collection of the Zodiac Gatekeys.
  Fact 0.187: Yukino states that when the twelve Zodiac Gatekeys are all gathered, a gateway that changes the world will open.
  Fact 0.188: Lucy turns down the offer.
  Fact 0.189: Milliana tells Erza that Kagura's sword, Archenemy, is intended to kill Jellal.
  Fact 0.190: Milliana tells Erza that she and Kagura share the same resentment towards Jellal.
  Fact 0.191: Natsu and Happy run after Yukino in an attempt to apologize.
  Fact 0.192: Natsu realizes that Yukino had been expelled.
  Fact 0.193: Natsu infiltrates the Crocus Garden to fight Jiemma.
  Fact 0.194: Episode 166, 'Pandemonium', has the Japanese title 'Pandemoniumu' and originally aired on January 26, 2013.
  Fact 0.195: Episode 166 aired in English on September 1, 2015.
  Fact 0.196: Natsu punches out Dobengal, the guild member summoned to fight him.
  Fact 0.197: Natsu and Jiemma engage in a duel.
  Fact 0.198: The guild master's daughter, Minerva, appears and ends their duel.
  Fact 0.199: Minerva takes Happy hostage.
  Fact 0.200: Minerva convinces Natsu to settle their dispute in the Grand Magic Games.
  Fact 0.201: As Natsu leaves, he states that Saber Tooth will never surpass Fairy Tail because they care for their friends.
  Fact 0.202: Rogue echoes Natsu's words.
  Fact 0.203: Sting is left in awe at Natsu's strength.
  Fact 0.204: The following day, an event called 'Pandemonium' begins.
  Fact 0.205: In Pandemonium, participants enter a magical citadel filled with 100 monsters of varying strengths.
  Fact 0.206: Participants individually do battle with any number of monsters they choose.
  Fact 0.207: Erza, representing Fairy Tail Team A, requests the 100 monsters as her opponents.
  Fact 0.208: Episode 167, '100 Against 1', has the Japanese title '100 tai 1' and originally aired on February 2, 2013.
  Fact 0.209: Episode 167 aired in English on September 1, 2015.
  Fact 0.210: Erza kills the 100 monsters using various types of armor to match their strengths and weaknesses.
  Fact 0.211: The crowd praises Fairy Tail.
  Fact 0.212: Fairy Tail is awarded 10 points.
  Fact 0.213: The game officials decide to determine the ranks of the remaining participants by means of a device called the 'Magic Power Finder', sometimes abbreviated as MPF.
  Fact 0.214: Participants must attack the Magic Power Finder with their magic power.
  Fact 0.215: The Magic Power Finder calculates and displays participants' equivalent scores based on the extent of power used.
  Fact 0.216: Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis.
  Fact 0.217: Makarov is surprised by Cana's use of the Fairy Glitter spell.
  Fact 0.218: Fairy Tail Team B receives 8 points.
  Fact 0.219: The audience bursts into a resounding swarm of cheers and claps due to the finish of the two Fairy Tail teams.
  Fact 0.220: The masked leader Alexei tells his subordinates Flare, Obra, Nalpudding, and Kurohebi that it is about time to proceed with their true mission.
  Fact 0.221: Raven Tail somehow got insider information regarding the battle matchups for the day.
  Fact 0.222: Episode 168, 'Laxus vs. Alexei', has the Japanese title 'Rakusasu vs. Arekusei' and originally aired on February 9, 2013.
  Fact 0.223: Episode 168 aired in English on September 1, 2015.
  Fact 0.224: In the battle portion of the Grand Magic Games' third day, Milliana of Mermaid Heel beats Semmes of Quatro Puppy.
  Fact 0.225: Saber Tooth's Rufus swiftly defeats Blue Pegasus' Eve.
  Fact 0.226: Laxus of Fairy Tail Team B is chosen to battle against Alexei of Team Raven Tail.
  Fact 0.227: The Fairy Tail members take extra precautions because Raven Tail is a suspicious guild.
  Fact 0.228: Fairy Tail watches the actions of Ivan Dreyar and the other four Raven Tail members.
  Fact 0.229: The spectators are shocked to see Laxus being beaten by Alexei.
  Fact 0.230: No cautious movements can be seen from the Raven Tail team.
  Fact 0.231: The two dueling people are illusions created by Alexei.
  Fact 0.232: Alexei created the illusions so that he could have a talk with Laxus.
  Fact 0.233: The other Raven Tail members being monitored by Fairy Tail are lifelike illusions.
  Fact 0.234: In reality, the other Raven Tail members are with Alexei on the battlefield.
  Fact 0.235: Alexei reveals himself to be Ivan Dreyar.
  Fact 0.236: Ivan and the Raven Tail members are unable to interrogate Laxus about the Lumen Histoire's location.
  Fact 0.237: Ivan and the Raven Tail members are forced to fight Laxus.
  Fact 0.238: Before their battle begins, Ivan declares that Raven Tail is an anti-Fairy Tail guild.
  Fact 0.239: Episode 169, 'Wendy vs. Shelia', has the Japanese title 'Wendi vs. Sheria' and originally aired on February 16, 2013.
  Fact 0.240: Episode 169 aired in English on September 1, 2015.
  Fact 0.241: Laxus defeats Ivan and the four Raven Tail elite members.
  Fact 0.242: Ivan's illusions disappear to the shock of everyone in the stadium.
  Fact 0.243: The game officials declare Laxus the victor.
  Fact 0.244: Ivan tells Laxus that Lumen Histoire is Fairy Tail's 'darkness'.
  Fact 0.245: Raven Tail is disqualified for the remainder of the Games due to the infraction of the rules.
  Fact 0.246: Raven Tail's members are captured by the soldiers.
  Fact 0.247: Wendy of Fairy Tail Team A and Sherria Blendy of Team Lamia Scale face off for the final match of the day.
  Fact 0.248: Sherria is revealed to be a Sky God Slayer.
  Fact 0.249: Gusts of wind surround the arena as Sherria and Wendy exchange attacks.
  Fact 0.250: Wendy employs an advanced Sky Dragon Slayer attack, much to Poluchka's astonishment.
  Fact 0.251: Sherria falls on the ground.
  Fact 0.252: Wendy is about to be announced as the winner of the bout.
  Fact 0.253: Sherria suddenly stands up with her wounds fully healed.
  Fact 0.254: The Crime Sorciere guild senses the nefarious aura.
  Fact 0.255: Jellal proceeds to the Domus Flau audience.
  Fact 0.256: Jellal is oblivious that he bumped into Doranbolt.
  Fact 0.257: Doranbolt wonders what Mystogan is doing in Earthland.
  Fact 0.258: Episode 170, 'Small Fists', has the Japanese title 'Chiisana Kobushi' and originally aired on February 23, 2013.
  Fact 0.259: Episode 170 aired in English on September 1, 2015.
  Fact 0.260: Cheria is fully healed.
  Fact 0.261: Wendy is almost depleted of magic power.
  Fact 0.262: Cheria casts a high-level spell towards Wendy.
  Fact 0.263: Wendy enhances Cheria's physical strength, causing the attack to miss Wendy.
  Fact 0.264: The two competitors engage in hand-to-hand combat.
  Fact 0.265: Due to the time limit and neither side giving up, their battle results in a draw.
  Fact 0.266: The draw gives both teams 5 points.
  Fact 0.267: Jellal tracks the sinister aura among the crowd in the form of a mysterious, hooded person.
  Fact 0.268: Jellal is about to confront the aura source.
  Fact 0.269: Doranbolt, Lahar, and other Rune Knights corner Jellal.
  Fact 0.270: Doranbolt interrogates Jellal.
  Fact 0.271: Jellal's cover is blown.
  Fact 0.272: The Rune Knights, Kagura and Milliana see the true identity of Mystogan.
  Fact 0.273: Yajima convinces the Rune Knights that Mystogan is Jellal's Edolas counterpart.
  Fact 0.274: Lahar releases the fugitive.
  Fact 0.275: Laxus and Makarov have a small conversation regarding Lumen Histoire.
  Fact 0.276: Mavis appears and explains that Lumen Histoire is the 'light' of Fairy Tail.
  Fact 0.277: Episode 171, 'Naval Battle', has the Japanese title 'Nabaru Batoru' and originally aired on March 2, 2013.
  Fact 0.278: Episode 171 aired in English on September 1, 2015.
  Fact 0.279: The jubilant Fairy Tail members celebrate their victories.
  Fact 0.280: Sting, Kagura, and Milliana are contemplating their own concerns.
  Fact 0.281: Yukino is enlisted as a sergeant under Arcadios' command.
  Fact 0.282: The fourth day of the competitions begins with an event called 'Naval Battle'.
  Fact 0.283: Naval Battle involves participants swimming in a giant water sphere and knocking the other competitors out.
  Fact 0.284: After the other participants fall out of the sphere, a five-minute battle between Lucy and Minerva ensues.
  Fact 0.285: Minerva quickly attacks Lucy and steals the keys of the Celestial Spirit mage.
  Fact 0.286: Lucy refuses to surrender.
  Fact 0.287: After hearing Lucy's intentions, Minerva stops attacking until the five-minute mark is reached.
  Fact 0.288: Minerva resumes attacking Lucy until the game officials decide to halt the match.
  Fact 0.289: The Fairy Tail members snap.
  Fact 0.290: Minerva dangles Lucy out of the sphere.
  Fact 0.291: The other Fairy Tail members rush to Lucy's aid.
  Fact 0.292: Episode 172, 'A Parfum for You', has the Japanese title 'Kimi ni Sasageru Parufamu' and originally aired on March 9, 2013.
  Fact 0.293: Episode 172 aired in English on September 1, 2015.
  Fact 0.294: Lucy is hospitalized and taken to the infirmary.
  Fact 0.295: The Fairy Tail members warn Saber Tooth that they have angered the worst possible guild to do so.
  Fact 0.296: Makarov announces that the two Fairy Tail teams must merge.
  Fact 0.297: The merger is a result of the disqualification of Raven Tail.
  Fact 0.298: The merger compensates for the odd number of teams for the battle round.
  Fact 0.299: The new team will have the former Team A's 35 points to start with.
  Fact 0.300: The new Fairy Tail team consists of Natsu, Gray, Erza, Gajeel, and Laxus.
  Fact 0.301: The new team is well-received as they enter the field.
  Fact 0.302: The second part of the tournament begins with a tag battle between Blue Pegasus' Ichiya and 'Rabbit' and Quatro Puppy's Rocker and Bacchus.
  Fact 0.303: 'Rabbit' is revealed to be Ichiya in disguise.
  Fact 0.304: Bacchus punches the Exceed.
  Fact 0.305: Ichiya bulks up to defeat the Quatro Puppy opponents.
  Fact 0.306: Blue Pegasus is awarded 10 points.
  Fact 0.307: Episode 173, 'Battle of Dragon Slayers!', has the Japanese title 'Batoru obu Doragon Sureiya' and originally aired on March 16, 2013.
  Fact 0.308: Episode 173 aired in English on September 1, 2015.
  Fact 0.309: Mermaid Heel's Kagura and Milliana face off against Lamia Scale's Lyon and Yuka for the second tag battle of the games.
  Fact 0.310: The second tag battle ends in a draw due to the time limit.
  Fact 0.311: For the third and final match of the day, Natsu and Gajeel of Team Fairy Tail enter the field against Sting and Rogue of Team Saber Tooth.
  Fact 0.312: Sting and Rogue were about to take the initiative when Natsu and Gajeel attacked them.
  Fact 0.313: The audience watches in shock as the Fairy Tail Dragon Slayers overpower the Saber Tooth Dragon Slayers.
  Fact 0.314: Sting activates 'White Drive'.
  Fact 0.315: Rogue powers up using 'Shadow Drive'.
  Fact 0.316: Episode 174, 'Four Dragons', has the Japanese title 'Yonin no Doragon' and originally aired on March 23, 2013.
  Fact 0.317: Episode 174 aired in English on September 1, 2015.
  Fact 0.318: Sting and Rogue begin attacking Natsu and Gajeel with their powers increased.
  Fact 0.319: After receiving heavy blows from the Saber Tooth Dragon Slayers, Natsu and Gajeel manage to hold them off.
  Fact 0.320: Sting and Rogue are forced to activate 'Dragon Force' on their own will.
  Fact 0.321: The Fairy Tail audience is shocked by Sting and Rogue activating 'Dragon Force'.
  Fact 0.322: Sting decides to finish the Fairy Tail Dragon Slayers alone.
  Fact 0.323: Sting held a promise to his Exceed friend Lector many years ago.
  Fact 0.324: The tide turns in favor of Sting as he unleashes a powerful attack on Natsu and Gajeel.
  Fact 0.325: Sting's attack causes the collapse of the arena floor.
  Fact 0.326: After a hand-to-hand combat underground, Sting is seemingly the victor of the match.
  Fact 0.327: Natsu and Gajeel stand up, comment on Sting's strength, and argue about his position while attacking.
  Fact 0.328: Natsu drags Gajeel on a mining cart and sends it away.
  Fact 0.329: Gajeel is left incapacitated due to motion sickness.
  Fact 0.330: Natsu challenges the Twin Dragons of Saber Tooth alone, in retribution for what Sting did earlier.
  Fact 0.331: Episode 175, 'Natsu vs. the Twin Dragons', has the Japanese title 'Natsu vs. Soryu' and originally aired on March 30, 2013.
  Fact 0.332: Episode 175 aired in English on September 1, 2015.
  Fact 0.333: The tag battle of the Dragon Slayers comes to a close.
  Fact 0.334: Natsu overpowers the combined attacks of Sting and Rogue.
  Fact 0.335: The Saber Tooth members watch in dismay.
  Fact 0.336: Jiemma seethes in anger over the defeat of the Twin Dragons.
  Fact 0.337: Jiemma's anger frightens the nearby spectators away.
  Fact 0.338: The rest of Fairy Tail celebrates in joy as its team receives 10 points.
  Fact 0.339: Fairy Tail surpasses Team Saber Tooth by a single point.
  Fact 0.340: Gajeel, back to his senses after being dragged by Natsu into a cart earlier, stumbles upon a cave full of dragon skeletons.
  Fact 0.341: That night, Jellal manages to corner the source of the dark aura.
  Fact 0.342: Jellal stands in disbelief at the identity of the mysterious, hooded person.

Retrieved items:
<item 1>
List of Fairy Tail episodes (season 6)
The three ask Fairy Tail for their help in determining the source of a magical anomaly similar to Zeref 's magic that they have sensed around the Grand Magic
Games arena , Domus Flau . In exchange , Ultear uses her magic to invoke their Second Origin in order to double their magic power . As the others undergo the
painful process of awakening their Second Origin , Erza talks with Jellal over his regained memories . The two nearly kiss , but Jellal decides that he and Erza
can not be together and lies to her that he has a fiancée . Erza quickly picks up on this and respects his decision , seeing Crime Sorcière off as they leave .
155 `` Crocus , the Flower - Blooming Capital '' `` Hanasaku Miyako : Kurokkasu '' ( 花 笑 く 都 ・ クロッカス ) November 3 , 2012 December 15 , 2015 Natsu , Lucy , Gray
, Erza , and Wendy are chosen to represent their guild in the Grand Magic Games , which are to be held in Fiore 's capital , Crocus .
</item 1>

<item 2>
List of Fairy Tail episodes (season 6)
Later , Jellal discusses the fact that he has not sensed the evil power similar to Zeref 's even though the second day of the Grand Magic Games has ended . Erza
reunites with Milliana , who is revealed to be a member of Mermaid Heel . Meanwhile , Saber Tooth 's master Jiemma is dissatisfied with his guild 's performance
. Though Sting is given another chance , Jiemma humiliates Yukino for her failure . 165 `` Hatred at Nightfall '' `` Urami wa Yoru no Tobari ni Tsutsumarete ''
( 怨み は 夜 の 帐 に 包 まれ て ) January 19 , 2013 September 1 , 2015 After Yukino is expelled from Saber Tooth , she offers her two golden Gatekeys to Lucy to complete
the latter 's collection of the Zodiac Gatekeys , stating that when the twelve Zodiac Gatekeys are all gathered , a gateway that changes the world will open ;
however , Lucy turns down the offer .
</item 2>

<item 3>
List of Fairy Tail episodes (season 6)
When the wizards arrive , they instead find a surprise party held by the Celestial Spirit King celebrating their return from Sirius Island after seven years .
They spend the day in the spirit world partying with the spirits . Before they leave , however , Virgo informs them that one day in the spirit world equals
three months in the human world , and they find the Grand Magic Games are upon them . 154 `` For All the Time We Missed Each Other '' `` Surechigatta Jikan no
Bun dake '' ( すれ 违 っ た 时间 の 分 だけ ) October 27 , 2012 December 8 , 2015 While mulling over their lost training time , the Fairy Tail wizards receive a message
via carrier pigeon requesting a private meeting in the forest nearby . There they meet a runaway Jellal , Ultear , and Merudy , who have formed their own guild
, Crime Sorcière , to defeat Zeref and any dark guilds that try to use his power .
</item 3>

<item 4>
List of Fairy Tail episodes (season 6)
As they tour the city , Natsu and Happy encounter the twin dragon slayers , Sting and Rogue , who mock Natsu for not being able to kill Acnologia , and boast of
becoming true dragon slayers by killing the dragons who raised them . At midnight , Fairy Tail and the other participating guilds are put through a preliminary
event -- a race through a massive floating labyrinth -- designed to narrow the number of participating guilds down to eight teams . 156 `` Sky Labyrinth '' ``
Sukai Rabirinsu '' ( 空中 迷宮 ( スカイ ラビリンス ) ) November 10 , 2012 December 22 , 2015 Natsu 's team notices that Wendy has gone missing , so Elfman replaces her to
keep them from being eliminated . As they make their way through the enchanted labyrinth , the five encounter various other guilds and steal the maps made by
the other guilds to better navigate the maze . They manage to reach the end of the maze and pass the elimination round , confident they have arrived first ,
only to discover they have barely taken last place .
</item 4>

<item 5>
List of Fairy Tail episodes (season 6)
In the end , Cana of Fairy Tail Team B dominates the event by using the Fairy Glitter spell lent to her by Mavis , much to Makarov 's surprise . With Cana 's
victory , Fairy Tail Team B receives 8 points , and the audience began to burst once more into a resounding swarm of cheers and claps due to the finish of the
two Fairy Tail teams . Meanwhile , the Raven Tail team somehow managed to get an insider information regarding the battle matchups for the day , so the masked
leader Alexei tells his subordinates Flare , Obra , Nalpudding , and Kurohebi that it 's about time to proceed with their true mission . 168 `` Laxus vs. Alexei
'' `` Rakusasu vs. Arekusei '' ( ラクサス vs. アレクセイ ) February 9 , 2013 September 1 , 2015 In the battle portion of the Grand Magic Games ' third day , Milliana of
Mermaid Heel beats Semmes of Quatro Puppy while Saber Tooth 's Rufus swiftly defeats Blue Pegasus ' Eve .
</item 5>

<item 6>
List of Fairy Tail episodes (season 6)
The following day as the Grand Magic Games continue , an event called `` Pandemonium '' begins . It involves the participants entering a magical citadel filled
with 100 monsters of varying strengths and individually doing battle with any number of monsters they choose . Representing Fairy Tail Team A , Erza requests
the 100 monsters as her opponents . 167 `` 100 Against 1 '' `` Hyaku tai Ichi '' ( 100 対 1 ) February 2 , 2013 September 1 , 2015 Erza kills the 100 monsters
using various types of armor to match their strength and weaknesses . As a result , the crowd praises Fairy Tail , and the guild is awarded with 10 points .
Moreover , the game officials decided to determine the ranks of the remaining participants by means of a device called `` Magic Power Finder '' ( sometimes
abbreviated as MPF ) . Using this tool , participants must attack it with their magic power , which will , in turn , calculate and display their equivalent
scores depending on the extent of power used .
</item 6>

<item 7>
List of Fairy Tail episodes (season 6)
Poluchka reveals herself as the Edolas counterpart of Grandine , and gives Wendy written instructions for two advanced spells that Grandine failed to teach her
. Meanwhile , Makarov announces Gildarts ' succession as master to the rest of the guild , only to find he has gone on a journey , leaving a letter of
resignation behind . In the letter , Gildarts reinstates Laxus into the guild , returns mastership to Makarov , and tells him to make Fairy Tail the strongest
guild in Fiore again . Romeo suggests they participate in the Grand Magic Games , an annual festival designed to determine the guild rankings in Fiore . 153 ``
Song of the Stars '' `` Hoshiboshi no Uta '' ( 星々 の 歌 ) October 20 , 2012 December 1 , 2015 With the Grand Magic Games occurring in three months time , Natsu
and his friends train at a beach resort . On their second day of training , Virgo appears and tells the wizards that the Celestial Spirit World is in danger of
destruction .
</item 7>

<item 8>
List of Fairy Tail episodes (season 6)
It contains a single story arc , titled `` Dai Matō Enbu '' ( 大 魔 闘 演武 , lit `` Grand Magic Games '' ) , which adapts material from the beginning of the 31st
through the ending of the 35th volumes of Hiro Mashima 's Fairy Tail manga over 24 episodes . Focusing on Natsu and the others who were frozen in time for seven
years on Tenrou Island , the members return to Magnolia and discover that Saber Tooth is the strongest guild in the Fiore Kingdom . Having become the weakest
guild during the absence of its core members , Fairy Tail decides to enter the Grand Magic Games , an annual competition to decide Fiore 's strongest guild .
The season began airing on October 6 , 2012 on TV Tokyo in Japan and concluded its run on March 30 , 2013 . The first DVD compilation , containing the last two
episodes of the fifth season and the first two episodes of the sixth season , was released by Pony Canyon on March 6 , 2013 .
</item 8>

<item 9>
List of Fairy Tail episodes (season 6)
To everyone 's shock , the next team is revealed to be a second team from Fairy Tail composed of Gajeel , Juvia , Mirajane , Laxus , and Jellal ( disguised as
Mystogan to help investigate the arena ) . The final guild to be announced is Saber Tooth , much to the joy of the audience . With all eight teams announced ,
they each send out one member to participate in the first contest of the games , with Gray and Juvia representing their teams . 158 `` Night of Shooting Stars
'' `` Hoshi Furu Yoru ni '' ( 星 降 ル 夜 ニ ) November 24 , 2012 January 5 , 2016 The games ' first contest , `` Hidden '' , requires the eight participants to
score points by finding and attacking each other in a virtual city filled with copies of themselves . Raven Tail member Nalpudding repeatedly targets Gray , who
scores last place in the contest while Saber Tooth 's Rufus scores first by using his Memory - Make magic to attack all the other participants at once .
</item 9>

<item 10>
List of Fairy Tail episodes (season 6)
Meanwhile , the fourth day of the competitions comes to a start with an event , `` Naval Battle '' , which involves the participants swimming in a giant water
sphere and knocking the other competitors out . After the other participants fall out of the aquatic sphere , a five - minute battle between Lucy and Minerva
ensues . However , Minerva quickly attacks Lucy , and she manages to steal the keys of the Celestial Spirit mage . Regardless of this , Lucy refuses to
surrender . After hearing Lucy 's intentions , Minerva stops from attacking up to the extent that the five - minute mark was reached . However , Minerva resumes
attacking Lucy until the game officials decide to halt the match . The Fairy Tail members snap , and Minerva dangles Lucy out of the sphere while the others
rush to her aid .
</item 10>
=== END ITEM id=689fe10b0a0502d5 ===

=== ITEM id=db41baa6f54fe0ff ===
Gold sentences and facts:
Sentence 0: `` The Gift of the Magi '' is a short story , written by O. Henry ( a pen name for William Sydney Porter ) , about a young husband and wife and how they deal with the challenge of buying secret Christmas gifts for each other with very little money . As a sentimental story with a moral lesson about gift - giving , it has been a popular one for adaptation , especially for presentation at Christmas time . The plot and its twist ending are well - known , and the ending is generally considered an example of comic irony . It was allegedly written at Pete 's Tavern on Irving Place in New York City .
  Fact 0.0: The Gift of the Magi is a short story.
  Fact 0.1: The Gift of the Magi was written by O. Henry.
  Fact 0.2: O. Henry is a pen name for William Sydney Porter.
  Fact 0.3: The Gift of the Magi is about a young husband and wife.
  Fact 0.4: The husband and wife deal with the challenge of buying secret Christmas gifts for each other with very little money.
  Fact 0.5: The Gift of the Magi is a sentimental story with a moral lesson about gift-giving.
  Fact 0.6: The Gift of the Magi has been a popular story for adaptation.
  Fact 0.7: The Gift of the Magi has been especially popular for presentation at Christmas time.
  Fact 0.8: The plot of The Gift of the Magi is well-known.
  Fact 0.9: The twist ending of The Gift of the Magi is well-known.
  Fact 0.10: The ending of The Gift of the Magi is generally considered an example of comic irony.
  Fact 0.11: The Gift of the Magi was allegedly written at Pete's Tavern on Irving Place in New York City.

Retrieved items:
<item 1>
The Gift of the Magi
LiteraryTraveler.com . Jump up ^ Не буду гангстером , дорогая Jump up ^ Mickey 's Once Upon A Christmas on IMDb Jump up ^ O. Henry on IMDb Jump up ^ Raincoat on
IMDb Jump up ^ The Gift of the Magi ( 2014 ) - IMDb Jump up ^ Darovete Na Vlahvite ( 2013 ) - IMDb Jump up ^ St. Germain , Mark ; Courts , Randy ( 1984 ) . The
Gifts of the Magi . New York : Dramatists Play Service , Inc . ISBN 978 - 0 - 8222 - 1461 - 8 Jump up ^ Watch A Trump Christmas from Saturday Night Live on
NBC.com , retrieved 2017 - 08 - 03 External links ( edit ) Wikisource has original text related to this article : The Gift of the Magi `` The Gift of the Magi
'' by O.
</item 1>

<item 2>
The Gift of the Magi
Unusually late , Jim walks in and immediately stops short at the sight of Della , who had previously prayed that she was still pretty to Jim . Della then admits
to Jim that she sold her hair to buy him his present . Jim gives Della her present -- an assortment of combs , useless now that her hair is short . Della then
shows Jim the chain she bought for him , to which Jim says he sold his watch to get the money to buy her combs . Although Jim and Della are now left with gifts
that neither one can use , they realize how far they are willing to go to show their love for each other , and how priceless their love really is . The story
ends with the narrator comparing the pair 's mutually sacrificial gifts of love with those of the Biblical Magi .
</item 2>

<item 3>
Graph query row:
entity: O. Henry
relationship: CREATED
related: The Gift of the Magi
detail: None
</item 3>

<item 4>
The Gift of the Magi
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 4>

<item 5>
The Gift of the Magi
The Gift of the Magi - wikipedia The Gift of the Magi Jump to : navigation , search This article is about the short story . For the Magi that visited baby Jesus
, see Biblical Magi . For other uses , see The Gift of the Magi ( disambiguation ) . `` The Gift of the Magi '' Author O. Henry Country United states Language
English Genre ( s ) Short story Published in The Four Million Publication type Anthology Publication date December 10 , 1905 ( newspaper ) ; April 10 , 1906 (
book ) `` The Gift of the Magi '' is a short story , written by O. Henry ( a pen name for William Sydney Porter ) , about a young husband and wife and how they
deal with the challenge of buying secret Christmas gifts for each other with very little money . As a sentimental story with a moral lesson about gift - giving
, it has been a popular one for adaptation , especially for presentation at Christmas time . The plot and its twist ending are well - known , and the ending is
generally considered an example of comic irony . It was allegedly written at Pete 's Tavern on Irving Place in New York City .
</item 5>

<item 6>
The Gift of the Magi
Adaptations ( edit ) The story has been adapted to films , The Sacrifice ( 1909 ) , Love 's Surprises Are Futile ( 1916 ) , The Gift of the Magi ( 1917 ) , a
segment of O. Henry 's Full House ( 1952 ) , The Gift of Love ( 1978 ) , The Gift of the Magi ( 1958 ) , Dary magów ( Poland , 1972 ) , Christmas Eve on Sesame
Street ( 1978 ) , I 'll not be a gangster , love ( Не буду гангстером , дорогая / Nebūsiu gangsteriu , brangioji , USSR , 1978 ) , Mickey 's Once Upon a
Christmas ( 1999 ) , The Gift of the Magi ( 2004 ) and the short film for the Irish band The Script in 2010 called For the First Time . Love , another French
movie , based some of its scenes on this story . Raincoat ( 2004 ) , a Hindi film directed by Rituparno Ghosh is an adaptation of the story .
</item 6>

<item 7>
The Gift of the Magi
The Greek film directed by Ismene Daskarolis ( 2014 ) places it in the economical crisis of Greece today . The Mexican film Nosotros los pobres includes this
tale as a small sub-plot . There is also a Bulgarian short film adaptation known as `` Darovete na vlahvite '' ( 2013 ) directed by Ivan Abadjiev . An off -
Broadway musical version entitled The Gifts of the Magi premiered at Lamb 's Theatre in New York City in 1984 . Written by Mark St. Germain and Randy Courts ,
the play is regularly produced in schools and regional theaters . In 1955 , in The Honeymooners episode 13 of season 1 , `` ' Twas the Night Before Christmas ''
involves Ralph looking for the perfect present for Alice for Christmas . Without money , he pawns his prized bowling ball to buy her an expensive gift , only to
find out in the end that she bought him a custom bowling ball bag .
</item 7>

<item 8>
The Gift of the Magi
Henry The Gift of the Magi public domain audiobook at LibriVox O'Henry's `` The Gift of the Magi '' : Common Core Exemplar from NEH 's EDSITEment `` The Gift of
the Magi '' study guide , teacher resources , themes , quotes ( hide ) O. Henry Short stories `` The Duplicity of Hargraves '' ( 1902 ) `` A Retrieved
Reformation '' ( 1903 ) `` The Cop and the Anthem '' ( 1904 ) `` The Gift of the Magi '' ( 1905 ) `` Conscience in Art '' ( 1907 ) `` The Last Leaf '' ( 1907 )
`` The Third Ingredient '' ( 1908 ) `` The Ransom of Red Chief '' ( 1910 ) `` Makes the Whole World Kin '' ( 1911 ) Novels Cabbages and Kings ( 1904 ) Story
collections The Four Million ( 1906 ) Roads of Destiny ( 1909 ) Whirligigs ( 1910 ) Waifs and Strays ( 1917 ) Related O. Henry Award William Sidney Porter House
O. Henry House and Museum O. Henry Hall O.
</item 8>

<item 9>
The Gift of the Magi
The special premiered on HBO on December 17 , 1978 . The special later aired on ABC in 1980 and on Nickelodeon in the 1990s . The special features several
original songs written by songwriter Paul Williams . The sketch comedy show Studio C contains a skit called `` Gift of the Magi '' in which Jim and Della argue
over their gift - giving . The television series My Little Pony : Friendship is Magic has the episode `` The Gift of the Maud Pie '' in which Pinkie Pie gives
Maud a perfect gift , costing her the confetti - using Party Cannon . Maud 's gift in return was confetti for the cannon of which Pinkie Pie had sold . Radio
drama series Adventures in Odyssey features a comedic adaptation of this story in its episode `` Gifts for Madge & Guy '' The animated sketch comedy series
Robot Chicken features a parody of the story in its fourth season , except that Jim does not sell his watch and instead buys Della lingerie , much to her
consternation . Joni Mitchell wrote and performed , but never recorded , a song based on the story .
</item 9>

<item 10>
The Gift of the Magi
It was allegedly written at Pete 's Tavern on Irving Place in New York City . The story was initially published in The New York Sunday World under the title ``
Gifts of the Magi '' on December 10 , 1905 . It was first published in book form in the O. Henry Anthology The Four Million in April 1906 . Contents ( hide ) 1
Summary 2 Adaptations 3 References 4 External links Summary ( edit ) Mr. James Dillingham Young ( `` Jim '' ) and his wife , Della , are a couple living in a
modest apartment . On Christmas Eve , with only $1.87 in hand , and desperate to find a gift for Jim , Della sells her hair for $20 to a nearby hairdresser
named Madame Sofronie , and eventually finds a platinum pocket watch fob chain for Jim 's watch for $21 . Satisfied with the perfect gift for Jim , Della runs
home and begins to prepare pork chops for dinner . At 7 o'clock , Della sits at a table near the door , waiting for Jim to come home .
</item 10>
=== END ITEM id=db41baa6f54fe0ff ===

=== ITEM id=193f5cef0ec7f1bc ===
Gold sentences and facts:
Sentence 0: The music video for `` Total Eclipse of the Heart '' was directed by Russell Mulcahy and was filmed on location at the Holloway Sanatorium , a large Victorian Gothic hospital near Virginia Water , Surrey , England . The video features Bonnie Tyler clad in white , dreaming or fantasizing about her students in a boys ' boarding school . Young men are seen dancing and participating in various school activities and singing in a choir .
  Fact 0.0: The music video for 'Total Eclipse of the Heart' was directed by Russell Mulcahy.
  Fact 0.1: The music video for 'Total Eclipse of the Heart' was filmed on location at the Holloway Sanatorium.
  Fact 0.2: The Holloway Sanatorium is a large Victorian Gothic hospital.
  Fact 0.3: The Holloway Sanatorium is near Virginia Water, Surrey, England.
  Fact 0.4: The music video features Bonnie Tyler.
  Fact 0.5: In the music video, Bonnie Tyler is clad in white.
  Fact 0.6: In the music video, Bonnie Tyler is dreaming or fantasizing about her students in a boys' boarding school.
  Fact 0.7: In the music video, young men are seen dancing.
  Fact 0.8: In the music video, young men are seen participating in various school activities.
  Fact 0.9: In the music video, young men are seen singing in a choir.

Retrieved items:
<item 1>
Total Eclipse of the Heart
Donald A. Guarisco , also from AllMusic , retrospectively reviewed Faster Than the Speed of Night , and noted the song as an `` epic ballad , '' describing the
whole album as `` rock at its most melodramatic . '' Jim Beviglia from American Songwriter said that Tyler 's raspy vocals helped to legitimize the `` melodrama
inherent in the lyrics , '' and described the song as a `` garment - rending , chest - beating ( and ) emotionally exhausting ballad '' that suits the throes of
a turbulent relationship . Music video ( edit ) Holloway Sanatorium steps , which feature in the video The music video for `` Total Eclipse of the Heart '' was
directed by Russell Mulcahy and was filmed on location at the Holloway Sanatorium , a large Victorian Gothic hospital near Virginia Water , Surrey , England .
The video features Bonnie Tyler clad in white , dreaming or fantasizing about her students in a boys ' boarding school . Young men are seen dancing and
participating in various school activities and singing in a choir .
</item 1>

<item 2>
Total Eclipse of the Heart
( Sunset Strippers Verse Club Mix ) `` Total Eclipse of the Heart '' ( Sunset Strippers Dub Mix ) `` Total Eclipse of the Heart '' ( Sunset Strippers Radio Edit
) Cover versions and impact in Mainstream Media ( edit ) In a 2013 UK survey , the song came first in a list of most popular songs to sing in the shower , above
songs by Justin Bieber , Robbie Williams , One Direction and Elton John . In 2015 the song was voted by the British public as the nation 's third favourite
1980s number one in a poll for ITV . `` Total Eclipse of the Heart '' received substantial media attention during the solar eclipse of March 20 , 2015 . Tyler
's version received a 214 % increase of Spotify streams throughout the day . A similar impact was experienced during the solar eclipse of August 21 , 2017 ,
when Nielsen Music reported a 503 % increase in record sales . Around that time , the song hit number one on the iTunes chart .
</item 2>

<item 3>
Total Eclipse of the Heart
Young men are seen dancing and participating in various school activities and singing in a choir . A long - running urban legend is that the boy who appears
throughout the video and who shakes Tyler 's hand at the end is former Italian footballer Gianfranco Zola . In a 2012 interview , Zola confirmed that he did not
appear in the video . Live performances ( edit ) Since the song 's release , Tyler performs `` Total Eclipse of the Heart '' in every concert she performs . ``
I sing it much better now than I used to , '' she told The Huffington Post . `` I think my voice is probably not as husky as it was , I think it 's mellowed a
bit . '' The song was performed at the 26th Annual Grammy Awards , held at the Shrine Auditorium , Los Angeles , on 28 February 1984 . Tyler performed a brief ,
live performance of this song on a Royal Caribbean cruise during the solar eclipse of 21 August 2017 . Live recordings of Tyler performing the song have been
released on her live albums Bonnie Tyler Live ( 2006 ) and Live in Germany 1993 ( 2011 ) .
</item 3>

<item 4>
Critics called “Total Eclipse of the Heart” an epic, melodramatic ballad, praising Bonnie Tyler’s raspy vocals and its portrayal of a turbulent relationship.
Russell Mulcahy’s video, filmed at Holloway Sanatorium, shows Tyler dreaming of boys at a boarding school; Gianfranco Zola later denied appearing in it. Tyler
performs the song at every concert and has performed it at the 1984 Grammys and during the 2017 solar eclipse. Live versions appear on
</item 4>

<item 5>
Total Eclipse of the Heart
He also hand - picked the recording band for the song . `` Total Eclipse of the Heart '' had to be shortened for radio play . Tyler did not believe that the
song was radio - friendly at its full length ; the song was reduced from seven minutes and two seconds to four minutes and thirty seconds . The power ballad
remains Tyler 's most successful song , peaking at No. 1 in the United States , South Africa , Australia , New Zealand , Canada and the United Kingdom . At its
peak , it sold 60,000 copies per day , and approximately 6 million copies in total . It won the Variety Club award in the UK for best single of 1983 . The song
also made number 82 of VH1 's top 100 love songs . `` Total Eclipse of the Heart '' A sample of `` Total Eclipse of the Heart '' , performed by Bonnie Tyler .
Problems playing this file ? See media help . Tyler told Record Mirror that she thought the song was about `` someone who wants to love so badly she 's lying
there in complete darkness . ''
</item 5>

<item 6>
Total Eclipse of the Heart
The song would have been released as the album 's second single but was cancelled due to their The Love Tour conflicts , though a promo release still reached
No. 5 on the radio charts in the Philippines . Three official remixes have been made for their version as well as a remix done by Jim Steinman , which was
ultimately rejected by the record label but has surfaced on the Internet . The song was released as promotional single on 2007 . Yuridia version ( edit )
Mexican singer Yuridia covered the song in Spanish as `` Eclipse Total del Amor '' from her second studio album Habla El Corazón in 2006 . The single peaked at
number 36 on the Billboard Latin Pop Songs chart . Parody versions ( edit ) A parody of the song and music video were published in 2009 , in what the fans and
makers call a `` literal video version '' , which is a type of video that replaces the original song lyrics with humorous lyrics describing the images in the
video . Time magazine listed it as the 6th best viral video of 2009 . This was also the 6th literal video produced by professional video editor David A. Scott
Jr.
</item 6>

<item 7>
The article’s references document “Total Eclipse of the Heart” chart information from sources including Apple, Ultratop, RPM, InfoDisc, GfK, the Irish Charts,
and Dutch chart services. The song is a 1983 Bonnie Tyler single written and produced by Jim Steinman, with a music video directed by Russell Mulcahy. It topped
charts in the US, UK, Australia, New Zealand, Norway, South Africa, Ireland, and Canada. The page also notes later versions
</item 7>

<item 8>
Total Eclipse of the Heart
; the singer who performed for this re-dub was Scott 's friend Felisha Noble using the pseudonym Persephone Maewyn . The video is still viewable on Funny or Die
's website , where it has received over 100,000 views , qualifying it for `` Immortal '' status since April 2012 . In 2010 , Tyler appeared in an advertisement
for MasterCard , performing a short parody of the song with its noted new lyric `` Turn around , Neville . '' She performed the original song in a similar
advertisement for Westpac in 2012 . Warby Parker produced a parody video in anticipation of the solar eclipse of August 21 , 2017 . See also ( edit ) 1983 in
British music List of best - selling singles List of Billboard Hot 100 number - one singles of 1983 List of number - one singles of 1983 ( Australia ) List of
number - one singles of 1983 ( Canada ) List of number - one singles of 1983 ( Ireland ) List of number - one singles in 1983 ( New Zealand ) References ( edit
) ^ Jump up to : `` Tyler releases new Total Eclipse '' .
</item 8>

<item 9>
Total Eclipse of the Heart
230 . Jump up ^ Scott , David ( 16 April 2011 ) . `` Total Eclipse of the Heart : Literal Video Version '' . Retrieved 26 August 2017 . Jump up ^ Benedictus ,
Leo ( 21 April 2010 ) . `` A word on our sponsors : Total eclipse of the card '' . The Guardian . Retrieved 20 August 2014 . Jump up ^ `` Westpac brings in
Bonnie Tyler to sing at couple 's fantasy wedding '' . mUmBRELLA. 21 November 2012 . Retrieved 22 August 2014 . Jump up ^ Matthewson , Samantha ( 1 August 2017
) . `` This ' Total Eclipse of the Heart ' Parody Video for the 2017 Solar Eclipse Is Just Awesome '' . Space.com . Retrieved 3 August 2017 .
</item 9>

<item 10>
After leaving RCA in 1981, Bonnie Tyler recruited manager David Aspden and asked Jim Steinman to produce her. Steinman and Rory Dodd performed “Total Eclipse of
the Heart” for her in 1982, and Steinman selected the recording band. The song’s popularity surged during the 2015 and 2017 solar eclipses, with major increases
in streams and sales. Covers included Westlife’s promotional version and Yuridia’s Spanish-language “Eclipse Total del Amor
</item 10>
=== END ITEM id=193f5cef0ec7f1bc ===

=== ITEM id=de0b06f8d01373ea ===
Gold sentences and facts:
Sentence 0: Although Surface Pro 4 is backward compatible with Type Cover accessory of Surface Pro 3 , an attachable keyboard which doubles as a screen protector , the Surface Pro 4 Type Cover has been announced with and without a fingerprint ID sensor . The new Type Cover is thinner and lighter than its predecessor , has improved magnetic stability for lap use , a mechanical keyboard with improved key spacing , as well as a larger touchpad .
  Fact 0.0: Surface Pro 4 is backward compatible with the Type Cover accessory of Surface Pro 3.
  Fact 0.1: The Type Cover is an attachable keyboard that doubles as a screen protector.
  Fact 0.2: The Surface Pro 4 Type Cover has been announced with a fingerprint ID sensor.
  Fact 0.3: The Surface Pro 4 Type Cover has been announced without a fingerprint ID sensor.
  Fact 0.4: The new Type Cover is thinner than its predecessor.
  Fact 0.5: The new Type Cover is lighter than its predecessor.
  Fact 0.6: The new Type Cover has improved magnetic stability for lap use.
  Fact 0.7: The new Type Cover has a mechanical keyboard with improved key spacing.
  Fact 0.8: The new Type Cover has a larger touchpad.

Retrieved items:
<item 1>
Surface Pro 4
The common traits , inherited from the previous generation device , are a continuous kickstand , magnesium casing with a silver - colored back side , 802.11 a /
b / g / n / ac Wi - Fi radio , Bluetooth 4.0 Low Energy , full - size USB 3.0 port , microSD card reader , headset jack , Mini DisplayPort , Cover Port , and a
renamed SurfaceConnect , a port on the side to connect a wall charger , Docking Station or Surface Dock accessories .
</item 1>

<item 2>
Surface Pro 4
com LinkedIn LinkExchange Lionhead Studios Maluuba Massive Incorporated Mobile Data Labs Mojang Nokia Devices and Services Onfolio Pando Networks Perceptive
Pixel PlaceWare Powerset ProClarity Rare Revolution Analytics ScreenTonic Secure Islands Simplygon Skype Sunrise Atelier SwiftKey Winternals Software Teleo
Telekinesys Research Tellme Networks Twisted Pixel Games Vermeer Technologies Visio Corporation Vivaty VoloMetrix VXtreme WebTV Networks Xamarin Yammer Yupi
Portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Surface_Pro_4&oldid=823914317 '' Categories : Microsoft Surface Tablet computers introduced
in 2015 2 - in - 1 PCs Talk Contents About Wikipedia فارسی 한국어 Türkçe 中文 Edit links This page was last edited on 4 February 2018 , at 05 : 05 . Text is
available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and
Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 2>

<item 3>
Surface Pro 4
designed , developed , marketed , and produced by Microsoft . The Surface Pro 4 was announced on October 6 , 2015 alongside the Surface Book . In the U.S. and
Canada , the Surface Pro 4 was released on October 26 , 2015 . The Surface Pro 4 is an update over its predecessor , featuring Skylake CPUs , more RAM and SSD
options and a 12.3 - inch display with a greater resolution . At the same time , the device is compatible with most of its predecessor 's accessories . Contents
( hide ) 1 History 2 Features 2.1 Hardware 2.2 Software 2.3 Accessories 3 Issues 4 Reception 5 Timeline 6 References History ( edit ) The Surface Pro 4 was
announced on October 6 , 2015 alongside the Surface Book . Both devices were available for pre-order the following day and available for customers beginning on
October 26 , 2015 . The device initially exhibited failure to sleep properly , which drained the battery very quickly . Microsoft subsequently developed a fix
that was made available on February 17 , 2016 .
</item 3>

<item 4>
Surface Pro 4
With Windows 10 the Tablet mode is available when the Type Cover is detached from the device . In this mode , all windows are opened full - screen and the
interface becomes more touch - centric . The device also has a Windows Hello - compatible camera . It combines the use of a regular and infrared cameras to
authenticate the user . Accessories ( edit ) The Surface Pro 4 is backward compatible with all accessories of its predecessor . At the Windows 10 Devices Event
on 6 October , 2015 , alongside new Surface 2 - in - 1s , new accessories were announced . The Surface Dock is a redesigned docking accessory in a brick form
factor . It is compatible with all past and future Surface models with a SurfaceConnect side port , previously used to connect a wall charger or Docking Station
accessory : Surface Pro 3 , 4 and Surface Book . The Surface Dock will have 2 Mini DisplayPorts , 1 Gigabit Ethernet , 4 USB 3.0 , and 1 audio out ports .
</item 4>

<item 5>
Surface Pro 4
Although Surface Pro 4 is backward compatible with Type Cover accessory of Surface Pro 3 , an attachable keyboard which doubles as a screen protector , the
Surface Pro 4 Type Cover has been announced with and without a fingerprint ID sensor . The new Type Cover is thinner and lighter than its predecessor , has
improved magnetic stability for lap use , a mechanical keyboard with improved key spacing , as well as a larger touchpad . An updated version of Surface Pen is
included with the Surface Pro 4 , which has 1024 levels of pressure sensitivity and is heavier with a rubberized tip . However , the new pen has only one button
on the side rather than the previous pen 's two . There is a eraser on the end that doubles as a button , which may perform different tasks , like opening the
OneNote app or activating Cortana , and can be customized using the Surface App . Issues ( edit ) The Surface Pro 4 is known to have wide spread flicker issues
with the display named as Flickergate after the independent website flickergate.com which was sat up by the affected users after getting no fix from Microsoft .
</item 5>

<item 6>
Surface Pro 4
256 GB , 512 GB , or 1 TB Display 12.3 inches ( 31 cm ) 2736x1824 ( 267ppi ) PixelSense display with 3 : 2 aspect ratio Graphics m3 : Intel HD graphics 515 i5 :
Intel HD graphics 520 i7 : Intel Iris 540 graphics 292.1 millimetres ( 11.50 in ) ( width ) 201.42 millimetres ( 7.930 in ) ( height ) 8.45 millimetres ( 0.333
in ) ( depth ) Weight m3 : 766 grams ( 1.689 lb ) i5 : 786 grams ( 1.733 lb ) i7 : 786 grams ( 1.733 lb ) Predecessor Surface Pro 3 Successor Surface Pro ( 5th
generation ) This article is part of a series on the Microsoft Surface Surface Surface Pro Pro 2 Pro 3 Pro 4 Pro ( 2017 ) Surface Book Surface Laptop Surface
Studio Surface Hub Accessories Surface Pen Covers Dock Dial The Surface Pro 4 is the fourth - generation Surface - series 2 - in - 1 detachable , designed ,
developed ,
</item 6>

<item 7>
Surface Pro 4
Microsoft 's spokesperson acknowledged the issue stating : `` We are aware that some customers have experienced a screen flicker on Surface Pro 4 and are
monitoring the situation closely . Customers impacted by this should contact Microsoft support . '' Microsoft has stated that bad batteries , pens and Type
Covers may be repaired under warranty . Reception ( edit ) The Surface Pro 4 received generally positive reviews from technology critics . Most praise a sharper
and larger display compared to the one found in the predecessor , redesigned cooling system , improved technical specifications , and an improved Type Cover .
The common disadvantages found by reviewers were that the Type Cover is still a separate purchase , the relatively modest performance of the Intel Core m3 chip
, and modest battery life . Timeline ( edit ) Source : Microsoft Devices Blog References ( edit ) Jump up ^ `` Microsoft Surface Pro 4 announced with new
Surface Pen , starts at $899 '' . The Verge . Jump up ^ `` Here are the details of the Surface Book and Surface Pro 4 chips , and why they matter '' . Jump up ^
`` ARK Intel Core m3 Specifications '' . Jump up ^ `` ARK Intel Core i5 Specifications '' .
</item 7>

<item 8>
Surface Pro 4
Surface Pro 4 - wikipedia Surface Pro 4 Jump to : navigation , search Surface Pro 4 Developer Microsoft Product family Surface Type 2 - in - 1 detachable
Generation 4th Release date October 26 , 2015 Introductory price $899 ( £ 749 ) Operating system Windows 10 Pro CPU Intel Skylake low - voltage dual - core : m3
- 6Y30 900 MHz up to 2.2 GHz , 4 MB cache , 4.5 W i5 - 6300U 2.4 up to 3.0 GHz , 3 MB cache , 15 W i7 - 6650U 2.2 up to 3.4 GHz , 4 MB cache , 15 W Memory 4 GB
, 8 GB , or 16 GB Storage PCI - Express SSD : 128 GB , 256 GB , 512 GB , or 1 TB Display 12.
</item 8>

<item 9>
Surface Pro 4
Microsoft subsequently developed a fix that was made available on February 17 , 2016 . Features ( edit ) Hardware ( edit ) The Surface Pro 4 is the same size as
the Surface Pro 3 , but has a thinner screen bezel which allows for a display of a slightly greater size of 12.3 inches ( 31 cm ) . The screen resolution is
also greater than the Surface Pro 3 's , at 2736x1824 at 267 PPI , with the same aspect ratio of 3 : 2 and 10 - point multi-touch . The chassis is 0.03 inches (
0.76 mm ) thinner and 0.03 pounds ( 14 g ) lighter than the Surface Pro 3 . The cooling system of the 2 - in - 1 has been redesigned and dubbed hybrid liquid
cooling system . It includes heat pipes with a flowing liquid , which is meant to avoid the use of the internal fan when the device is used for less demanding
tasks such as web browsing .
</item 9>

<item 10>
Surface Pro 4
( hide ) Surface Pro 4 Configuration Options Price Tier ( USD ) CPU Integrated GPU RAM Internal Storage 899 Intel Core m3 - 6Y30 ( 0.9 to 2.2 GHz ) HD 515 4 GB
128 GB 999 Intel Core i5 - 6300U ( 2.4 to 3.0 GHz ) HD 520 1299 8 GB 256 GB 1499 16 GB 1699 8 GB 512 GB 1899 16 GB 1599 Intel Core i7 - 6650U ( 2.2 to 3.4 GHz )
Iris 540 8 GB 256 GB 1799 16 GB 2199 512 GB 2699 1 TB Software ( edit ) All Surface Pro 4 models come with a 64 - bit version of Windows 10 Pro and a Microsoft
Office 30 - day trial . Windows 10 comes pre-installed with Mail , Calendar , People , Xbox ( app ) , Photos , Movies and TV , Groove , and Microsoft Edge .
With Windows 10 the Tablet mode is available when the Type Cover is detached from the device .
</item 10>
=== END ITEM id=de0b06f8d01373ea ===
