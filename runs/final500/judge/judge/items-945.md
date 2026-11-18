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

=== ITEM id=80434eb48a98800d ===
Gold sentences and facts:
Sentence 0: Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 . A sequel , Deadpool 3 , is in development .
  Fact 0.0: Deadpool 2 is scheduled to be released in the United States on May 18, 2018.
  Fact 0.1: A sequel, Deadpool 3, is in development.

Retrieved items:
<item 1>
Deadpool 2
BoxOffice . Retrieved March 23 , 2018 . Jump up ^ D'Alessandro , Anthony ( April 20 , 2018 ) . `` ' Deadpool 2 ' Advance Ticket Sales Records : Fandango 's Best
R - Rated Preseller Ever In First 24 Hours '' . Deadline Hollywood . Penske Business Media . Retrieved April 20 , 2018 . Jump up ^ D'Alessandro , Anthony (
April 26 , 2018 ) . `` ' Deadpool 2 ' Tracking Suggests Raunchy Superhero Headed For $150 M+ Opening '' . Deadline Hollywood . Penske Business Media . Retrieved
April 26 , 2018 . Jump up ^ Kit , Borys ( November 9 , 2016 ) . `` Fox 's X-Men Issues : Jennifer Lawrence Unsigned , ' Deadpool ' Defections , ' Gambit ' on
Hold '' . The Hollywood Reporter . Archived from the original on May 24 , 2017 . Retrieved May 24 , 2017 . Jump up ^ Eisenberg , Eric ( March 21 , 2017 ) .
</item 1>

<item 2>
Deadpool 2
Officially titling the film Deadpool 2 , the trailer also revealed that Crews was portraying Bedlam ; that Delaney was in the film , portraying Peter ; and that
Bill Skarsgård also had a role . The film 's final trailer , released a month later , confirmed that Lewis Tan would portray Shatterstar , joining Domino ,
Bedlam , and Peter as members of the X-Force in the film . Music Deadpool composer Junkie XL chose not to return for the sequel due to Miller 's departure . In
October 2017 , Tyler Bates was hired to write the score for the film . Release Deadpool 2 is set to be released in the United States on May 18 , 2018 . It was
previously scheduled for release on June 1 of that year . Marketing For the Fox presentation at CineEurope 2017 in June , Reynolds made a video message
featuring himself in costume as Deadpool from the film 's set . The first teaser poster , which pays homage to Norman Rockwell 's 1943 painting Freedom from
Want , was released that November .
</item 2>

<item 3>
Deadpool 2
Deadpool 2 - wikipedia Deadpool 2 Jump to : navigation , search Deadpool 2 Teaser poster Directed by David Leitch Produced by Simon Kinberg Ryan Reynolds Lauren
Shuler Donner Written by Rhett Reese Paul Wernick Ryan Reynolds Based on Deadpool by Rob Liefeld Fabian Nicieza Starring Ryan Reynolds Josh Brolin Morena
Baccarin Julian Dennison Zazie Beetz T.J. Miller Brianna Hildebrand Jack Kesy Stefan Kapičić Music by Tyler Bates Cinematography Jonathan Sela Edited by Craig
Alpert Elísabet Ronaldsdóttir Production company Marvel Entertainment The Donners ' Company Genre Films Distributed by 20th Century Fox Release date May 18 ,
2018 ( 2018 - 05 - 18 ) ( United States ) Country United States Language English Deadpool 2 is an upcoming American superhero film based on the Marvel Comics
character Deadpool , distributed by 20th Century Fox . It is intended to be the eleventh installment in the X-Men film series , and a sequel to the 2016 film
Deadpool .
</item 3>

<item 4>
Deadpool 2
During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident . Deadpool 2 is scheduled to be released in the United States on May 18 , 2018 .
A sequel , Deadpool 3 , is in development . Contents ( hide ) 1 Premise 2 Cast 3 Production 3.1 Development 3.2 Pre-production 3.3 Filming 3.4 Post-production 4
Music 5 Release 5.1 Marketing 6 Box office projection 7 Sequel 8 References 9 External links Premise Deadpool forms a team of mutants called the X-Force to
protect young mutant Russell from the time - traveling soldier Cable . Cast Ryan Reynolds as Wade Wilson / Deadpool : A wisecracking mercenary with accelerated
healing but severe scarring over his body after undergoing an experimental regenerative mutation . He forms the X-Force , a team of mutants . Josh Brolin as
Nathan Summers / Cable : A time travelling cybernetic mutant soldier , `` in many ways the opposite of Deadpool '' .
</item 4>

<item 5>
Deadpool 2
The film is directed by David Leitch from a script by Rhett Reese , Paul Wernick , and Ryan Reynolds , with Reynolds starring in the title role alongside Josh
Brolin , Morena Baccarin , Julian Dennison , Zazie Beetz , T.J. Miller , Brianna Hildebrand , Jack Kesy , and Stefan Kapičić . In Deadpool 2 , Deadpool forms
the team X-Force to protect a young mutant from Cable . Plans for a sequel to Deadpool began before that film 's release , and were confirmed in February 2016 .
Though the original creative team of Reynolds , Reese , Wernick , and director Tim Miller were quickly set to return for the second film , Miller left the
project in October 2016 due to creative differences with Reynolds , and was soon replaced by Leitch . An extensive casting search took place to fill the role of
Cable , with Brolin ultimately cast ; the casting of Beetz as Domino was also noteworthy . Filming took place in British Columbia , Canada , from June to
October 2017 . During filming , stunt woman Joi `` SJ '' Harris died in a motorcycle accident .
</item 5>

<item 6>
Deadpool 2
As part of a promotion for the film with beverage company Mike 's Harder , Sister Margaret 's School for Wayward Girls -- the fictional bar featured in the
films -- will be emulated in pop - ups at the Alligator Lounge in New York from April 26 -- 28 and the Slipper Clutch in Los Angeles from May 10 -- 12 . The pop
- ups will respectively serve pizza and chimichangas , traditionally Deadpool 's favorite food , as well as Deadpool - inspired Harder drinks . The campaign
also includes the chance to win a trip to the film 's premiere through Harder , and the Los Angeles pop - up will benefit the nonprofit DTLA Film Festival . Box
office projection Initial box office projections for the film in March 2018 expected it to gross $100 million over its opening weekend in the United States and
Canada , with a final domestic total expected to be lower than that of the first film ( $363 million ) at around $242 million . On April 20 , 2018 , both
Fandango and Regal Cinemas announced that Deadpool 2 was the best pre-selling R - rated film in their respective histories .
</item 6>

<item 7>
Deadpool 2
The pair confirmed that Stefan Kapičić 's Colossus , Brianna Hildebrand 's Negasonic Teenage Warhead , and Karan Soni 's Dopinder would be returning from the
first film to `` make at least an appearance '' , and that the film would not be exploring the details of Cable 's convoluted comic origins . By the next month
, it had been noted that there had been few announcements from Fox regarding the film , and that a release date had not yet been set which was seen as unusual
for sequels to popular films ( often studios `` announce sequels far too soon and with no idea what will happen '' ) , with the issue deemed to be the film 's
script . Reynolds , Reese , and Wernick had been `` bunkered down , working on the script together , trying to cross the finish line and create something
everyone is excited to make '' . At that time , Goddard joined the project to consult on the script . By the beginning of March 2017 , Michael Shannon had been
in the running to portray Cable , but no longer could due to a scheduling conflict .
</item 7>

<item 8>
Deadpool 2
By April 26 , opening weekend industry projections had reached upwards of $150 million . Sequel By November 2016 , with development underway on Deadpool 2 , Fox
was also planning Deadpool 3 , which was said to include the team X-Force . With the confirmation that Leitch would direct Deadpool 2 , it was revealed that Fox
was looking for a separate filmmaker to direct Deadpool 3 . In March 2017 , Reese clarified that though Deadpool 2 sets - up the X-Force team , a future film
focused on the team would be separate from Deadpool 3 , `` so I think we 'll be able to take two paths . ( X-Force ) is where we 're launching something bigger
, but then ( Deadpool 3 is ) where we 're contracting and staying personal and small . '' After the proposed acquisition of 21st Century Fox by Disney was
announced in December 2017 , Disney CEO Bob Iger said that Deadpool would be integrated with the Marvel Cinematic Universe under Disney , and that the company
would be willing to make future R - rated Deadpool films `` as long as we let the audiences know what 's coming '' .
</item 8>

<item 9>
Deadpool 2
Retrieved December 16 , 2017 . External links Official website Deadpool 2 on IMDb Deadpool Fabian Nicieza Rob Liefeld Supporting characters Agent X Avengers
Blind Al Bob , Agent of Hydra Sandi Brandenberg Cable Copycat Death Domino Emrys Killebrew Negasonic Teenage Warhead Outlaw Shiklah Siryn Spider - Man Weasel
Wolverine X-Men Antagonists Ajax Black Box Black Swan Black Talon Black Tom Cassidy Bullseye Doctor Bong Dracula Hit - Monkey Flag - Smasher Juggernaut Madcap
Sabretooth Slayback Taskmaster T - Ray Thanos Trapster Typhoid Mary Publications Cable & Deadpool Mrs. Deadpool and the Howling Commandos Wolverine and Deadpool
In other media Deadpool ( video game ) Deadpool ( film ) accolades box office records soundtrack No Good Deed Deadpool 2 Related articles 5 Ronin Gwenpool Mercs
for Money X-Force Deathstroke X-Men in other media Films X-Men X2 X-Men : The Last Stand soundtrack X-Men Origins : Wolverine X-Men : First Class The Wolverine
X-Men : Days of Future Past Deadpool soundtrack X-Men : Apocalypse Logan Deadpool 2 X-Men : Dark Phoenix The New Mutants Development of Gambit Characters
</item 9>

<item 10>
Deadpool 2
Josh Brolin emerged as a `` surprise contender '' to play Cable in April , ahead of Shannon and Harbour , and was officially cast in the role . Brolin also
portrays the Marvel Comics character Thanos , in the Marvel Cinematic Universe . Leslie Uggams confirmed that she would be reprising her role of Blind Al from
the first film , while Fox gave the sequel a June 1 , 2018 release date . Noting the release date , Leitch wanted to ensure that the film `` was worthy of a
summer tentpole movie , and we knew we were going to be wedged in between some big films '' , specifically wanting to expand the action and make the general
feeling of the film `` bigger '' than the first one . However , he wanted the film to have the same DNA as the original `` in terms of the tone , and the fun
... I love that challenge '' of combining that with the increased scope . The film uses Deadpool 's fourth wall - breaking to reference this release period ,
with Leitch calling these comments `` definitely fresh and timely when they come up . ''
</item 10>
=== END ITEM id=80434eb48a98800d ===

=== ITEM id=f89f987a9787ad2d ===
Gold sentences and facts:
Sentence 0: Large denominations of United States currency greater than $100 were circulated by the United States Treasury until 1969 . Since then , U.S. dollar banknotes have only been issued in seven denominations : $1 , $2 , $5 , $10 , $20 , $50 , and $100 .
  Fact 0.0: Large denominations of United States currency greater than $100 were circulated by the United States Treasury until 1969.
  Fact 0.1: Since 1969, U.S. dollar banknotes have only been issued in seven denominations.
  Fact 0.2: The seven denominations of U.S. dollar banknotes issued since 1969 are $1, $2, $5, $10, $20, $50, and $100.

Retrieved items:
<item 1>
Large denominations of United States currency
dollar U.S. Mint Philadelphia San Francisco Denver West Point Bureau of Engraving and Printing Mutilated currency Current coinage Penny ( 1 ¢ ) Nickel ( 5 ¢ )
Dime ( 10 ¢ ) Quarter ( 25 ¢ ) Half dollar ( 50 ¢ ) Dollar ( $1 ) Bullion coinage America the Beautiful Silver Bullion Coins American Buffalo American Gold
Eagle American Platinum Eagle American Silver Eagle American Palladium Eagle Current paper money $1 $2 $5 $10 $20 $50 $100 See also Bicentennial coinage
Commemoratives Early Commemoratives Modern Commemoratives Confederate dollar Large denominations Obsolete denominations Promotional fake denominations
Counterfeit United States currency Mill Coin production In God We Trust E pluribus unum Annuit cœptis Nicknames Replacement banknote Sales tax token ( hide )
Obsolete United States currency and coinage Topics United States coinage United States dollar History of the United States dollar Coins Half cent ( 1793 -- 1857
) Fugio Cent ( 1787 ) Large cent ( 1793 -- 1857 ;
</item 1>

<item 2>
Large denominations of United States currency
( 1995 ) . United States Large Size National Bank Notes . Society of Paper Money Collectors , Inc . ISBN 0 - 9648774 - 1 - 4 . Newman , Eric P. ( 2008 ) . The
Early Paper Money of America ( 5 ed . ) . Krause Publications . ISBN 0 - 89689 - 326 - X . Schwartz , John ; Lindquist , Scott ( 2011 ) . Standard Guide to
Small - Size U.S. Paper Money -- 1928 to Date . Krause Publications . ISBN 978 - 1 - 4402 - 1703 - 6 . Retrieved 14 February 2014 . External links ( edit )
Media related to Money of the United States by denomination at Wikimedia Commons Large Denominations from the Bureau of Engraving and Printing U.S. Department
of the Treasury United States currency and coinage Topics Federal Reserve System Federal Reserve Note U.S. dollar U.S.
</item 2>

<item 3>
Large denominations of United States currency
This was especially true for gold certificates from 1865 to 1934 . However , for the most part , the introduction of an electronic money system has made large -
scale cash transactions obsolete . When combined with concerns about counterfeiting and the use of cash in unlawful activities such as the illegal drug trade
and money laundering , it is unlikely that the U.S. government will re-issue large denomination currency in the near future , despite the amount of inflation
that has occurred since 1969 ( a $500 bill is now worth less , in real terms , than a $100 bill was worth in 1969 ) . According to the U.S. Department of
Treasury website , `` The present denominations of our currency in production are $1 , $2 , $5 , $10 , $20 , $50 and $100 . The purpose of the United States
currency system is to serve the needs of the public and these denominations meet that goal . Neither the Department of the Treasury nor the Federal Reserve
System has any plans to change the denominations in use today . ''
</item 3>

<item 4>
Large denominations of United States currency
See history of the United States dollar . Passive retirement ( edit ) Although they are still legal tender in the United States , high - denomination bills were
last printed on December 27 , 1945 , and officially discontinued on July 14 , 1969 , by the Federal Reserve System , due to ' lack of use ' . The $5,000 and
$10,000 effectively disappeared well before then . The Federal Reserve began taking high - denomination currency out of circulation ( destroying large bills
received by banks ) in 1969 . As of May 30 , 2009 , only 336 $10,000 bills were known to exist ; 342 remaining $5,000 bills ; and 165,372 remaining $1,000 bills
. Due to their rarity , collectors often pay considerably more than the face value of the bills to acquire them . Some are in museums in other parts of the
world . For the most part , these bills were used by banks and the Federal government for large financial transactions . This was especially true for gold
certificates from 1865 to 1934 .
</item 4>

<item 5>
Large denominations of United States currency
currency were issued in high - denomination notes across nearly 20 different series dates . The obverse of United States banknotes generally depict either
historical figures , allegorical figures symbolizing significant concepts ( e.g. , liberty , justice ) , or a combination of both . The reverse designs range
from abstract scroll - work with ornate denomination identifiers to reproductions of historical art works . Public versus institutional use ( edit ) Series 1934
gold certificates ( $100 , $1,000 , $10,000 and $100,000 ) were issued after the gold standard was repealed and gold was compulsorily confiscated by order of
President Franklin Roosevelt on March 9 , 1933 ( see United States Executive Order 6102 ) . Thus the series 1934 notes were used only for intragovernmental (
i.e. , Federal Reserve Bank ) transactions and were not issued to the public . This series was discontinued in 1940 . The series 1928 gold certificate reverse
was printed in black and green . See history of the United States dollar .
</item 5>

<item 6>
Large denominations of United States currency
1915 ) Patterns Two - cent billon ( 1836 ) Three - cent bronze ( 1863 ) Stella ( 1879 -- 1880 ) Currency Discontinued denominations Fractional currency Large
denominations of currency Discontinued currency types Early American currency Compound interest treasury note Demand Note Federal Reserve Bank Note Gold
certificate Interest bearing note National Bank Note National gold bank note Refunding Certificate Silver certificate Treasury or Coin Note Treasury Note ( 19th
century ) United States Note United States postal notes Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Large_denominations_of_United_States_currency&oldid=834172209 '' Categories : Paper money of the United States Hidden
categories : Articles needing additional references from February 2016 All articles needing additional references Articles containing potentially dated
statements from May 2009 All articles containing potentially dated statements Use mdy dates from November 2012 Talk Contents About Wikipedia فارسی Polski Tiếng
Việt 中文 Edit links This page was last edited on 4 April 2018 , at 09 : 16 . Text is available under the Creative Commons Attribution - ShareAlike License ;
additional terms may apply .
</item 6>

<item 7>
Large denominations of United States currency
In popular culture ( edit ) Before they were ' retired ' , host Monty Hall regularly tempted or teased contestants on Let 's Make a Deal with a $1,000 bill
featuring Grover Cleveland . High - denomination ( HD ) banknote issuing data ( edit ) Key to high denomination banknote type abbreviations Abbr Type Size
Series dates High denomination series date Comments $500 $1,000 $5,000 $10,000 $100,000 LT Legal tender Large 1862 -- 1923 1862 1863 1869 1874 1875 1878 1880
1862 1863 1869 1878 1880 1878 1878 -- CITN Compound interest treasury note Exception 1863 -- 1864 1863 1864 1864 -- -- -- IBN Interest bearing note Exception
1861 -- 1865 1861 1863 1864 1865 1861 1863 1864 1865 1861 1863 1864 1865 -- -- SC Silver certificate Large 1878 -- 1923 1878 1880 1878 1880
</item 7>

<item 8>
United States two-dollar bill
dollar U.S. Mint Philadelphia San Francisco Denver West Point Bureau of Engraving and Printing Mutilated currency Current coinage Penny ( 1 ¢ ) Nickel ( 5 ¢ )
Dime ( 10 ¢ ) Quarter ( 25 ¢ ) Half dollar ( 50 ¢ ) Dollar ( $1 ) Bullion coinage America the Beautiful Silver Bullion Coins American Buffalo American Gold
Eagle American Platinum Eagle American Silver Eagle American Palladium Eagle Current paper money $1 $2 $5 $10 $20 $50 $100 See also Bicentennial coinage
Commemoratives Early Commemoratives Modern Commemoratives Confederate dollar Large denominations Obsolete denominations Promotional fake denominations
Counterfeit United States currency Mill Coin production In God We Trust E pluribus unum Annuit cœptis Nicknames Replacement banknote Sales tax token Retrieved
from `` https://en.wikipedia.org/w/index.
</item 8>

<item 9>
Large denominations of United States currency
Large denominations of United States currency - wikipedia Large denominations of United States currency Jump to : navigation , search This article needs
additional citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and
removed . ( February 2016 ) ( Learn how and when to remove this template message ) Large denominations of United States currency greater than $100 were
circulated by the United States Treasury until 1969 . Since then , U.S. dollar banknotes have only been issued in seven denominations : $1 , $2 , $5 , $10 , $20
, $50 , and $100 . Contents ( hide ) 1 Overview and history 2 Public versus institutional use 3 Passive retirement 4 In popular culture 5 High - denomination (
HD ) banknote issuing data 6 High - denomination United States banknotes 7 See also 8 References 8.1 Footnotes 8.2 Notes 8.3 Bibliography 9 External links
Overview and history ( edit ) Large - denomination currency ( i.e.
</item 9>

<item 10>
Large denominations of United States currency
The display has since been dismantled and the bills sold to private collectors . Jump up ^ The following types of United States banknotes were not issued in
high denominations and are not included in the list below : Demand notes , Federal Reserve bank notes ( large or small size ) , legal tender ( small size ) ,
silver certificates ( small size ) , National Bank Notes ( small size ) The table sections are sorted by their appearance in the Friedberg reference book . Jump
up ^ Large size notes represent the earlier types or series of U.S. banknotes . Their `` average '' dimension is ​ 7 ⁄ × ​ 3 ⁄ inches ( 187 × 79 mm ) . Small
size notes ( described as such due to their size relative to the earlier large size notes ) are an `` average '' ​ 6 ⁄ × ​ 2 ⁄ ( 156 × 67 mm ) , the size of
modern U.S. currency . `` Each measurement is ± 0.08 inches ( 2 mm ) to account for margins and cutting '' .
</item 10>
=== END ITEM id=f89f987a9787ad2d ===

=== ITEM id=bfc8a7308e984ffd ===
Gold sentences and facts:
Sentence 0: The park lies in the north - east of South Africa , in the eastern parts of Limpopo and Mpumalanga provinces . Phalaborwa , Limpopo is the only town in South Africa that borders the Kruger National Park . It is one of the largest national parks in the world , with an area of 19,485 square kilometres ( 7,523 sq mi ) . The park is approximately 360 kilometres ( 220 mi ) long , and has an average width of 65 kilometres ( 40 mi ) . At its widest point , the park is 90 kilometres ( 56 mi ) wide from east to west .
  Fact 0.0: The Kruger National Park lies in the north-east of South Africa.
  Fact 0.1: The Kruger National Park lies in the eastern parts of Limpopo and Mpumalanga provinces.
  Fact 0.2: Phalaborwa, Limpopo is the only town in South Africa that borders the Kruger National Park.
  Fact 0.3: The Kruger National Park is one of the largest national parks in the world.
  Fact 0.4: The Kruger National Park has an area of 19,485 square kilometres (7,523 sq mi).
  Fact 0.5: The Kruger National Park is approximately 360 kilometres (220 mi) long.
  Fact 0.6: The Kruger National Park has an average width of 65 kilometres (40 mi).
  Fact 0.7: At its widest point, the Kruger National Park is 90 kilometres (56 mi) wide from east to west.
Sentence 1: Kruger National Park is one of the largest game reserves in Africa . It covers an area of 19,485 square kilometres ( 7,523 sq mi ) in the provinces of Limpopo and Mpumalanga in northeastern South Africa , and extends 360 kilometres ( 220 mi ) from north to south and 65 kilometres ( 40 mi ) from east to west . The administrative headquarters are in Skukuza . Areas of the park were first protected by the government of the South African Republic in 1898 , and it became South Africa 's first national park in 1926 .
  Fact 1.0: Kruger National Park is one of the largest game reserves in Africa.
  Fact 1.1: Kruger National Park covers an area of 19,485 square kilometres (7,523 sq mi).
  Fact 1.2: Kruger National Park is in the provinces of Limpopo and Mpumalanga.
  Fact 1.3: Kruger National Park is in northeastern South Africa.
  Fact 1.4: Kruger National Park extends 360 kilometres (220 mi) from north to south.
  Fact 1.5: Kruger National Park extends 65 kilometres (40 mi) from east to west.
  Fact 1.6: The administrative headquarters of Kruger National Park are in Skukuza.
  Fact 1.7: Areas of Kruger National Park were first protected by the government of the South African Republic in 1898.
  Fact 1.8: Kruger National Park became South Africa's first national park in 1926.
Sentence 2: Kruger National reserve IUCN category II ( national park ) Burchell 's zebra in a Kruger landscape Location of the park ( red area ) in South Africa Location Limpopo and Mpumalanga provinces , South Africa Nearest city Nelspruit ( southern ) Phalaborwa ( central ) Coordinates 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Coordinates : 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Area 19,485 km ( 7,523 sq mi ) Established 31 May 1926 ( 1898 ) Visitors 1,336,981 ( in 2004 ) Governing body South African National Parks www.sanparks.org/parks/kruger/
  Fact 2.0: Kruger National Reserve is IUCN category II (national park).
  Fact 2.1: Kruger National Reserve is located in Limpopo and Mpumalanga provinces, South Africa.
  Fact 2.2: The nearest city to Kruger National Reserve is Nelspruit (southern).
  Fact 2.3: The nearest city to Kruger National Reserve is Phalaborwa (central).
  Fact 2.4: Kruger National Reserve has coordinates 24°0′41''S 31°29′7''E / 24.01139°S 31.48528°E.
  Fact 2.5: Kruger National Reserve has an area of 19,485 km (7,523 sq mi).
  Fact 2.6: The infobox lists Kruger National Reserve as established on 31 May 1926, with 1898 given in parentheses.
  Fact 2.7: Kruger National Reserve had 1,336,981 visitors in 2004.
  Fact 2.8: Kruger National Reserve is governed by South African National Parks.
  Fact 2.9: The website for Kruger National Reserve is www.sanparks.org/parks/kruger/.

Retrieved items:
<item 1>
Kruger National Park
981 ( in 2004 ) Governing body South African National Parks www.sanparks.org/parks/kruger/ Kruger National Park is one of the largest game reserves in Africa .
It covers an area of 19,485 square kilometres ( 7,523 sq mi ) in the provinces of Limpopo and Mpumalanga in northeastern South Africa , and extends 360
kilometres ( 220 mi ) from north to south and 65 kilometres ( 40 mi ) from east to west . The administrative headquarters are in Skukuza . Areas of the park
were first protected by the government of the South African Republic in 1898 , and it became South Africa 's first national park in 1926 . To the west and south
of the Kruger National Park are the two South African provinces of Limpopo and Mpumalanga . In the north is Zimbabwe , and to the east is Mozambique . It is now
part of the Great Limpopo Transfrontier Park , a peace park that links Kruger National Park with the Gonarezhou National Park in Zimbabwe , and with the Limpopo
National Park in Mozambique .
</item 1>

<item 2>
Kruger National Park
16500 ° E ﻿ / - 23.94556 ; 31.16500 ﻿ ( Phalaborwa Gate ) Punda Maria Gate on the R524 road from Thohoyandou 22 ° 44 ′ 18 '' S 31 ° 0 ′ 33 '' E ﻿ / ﻿ 22.73833 °
S 31.00917 ° E ﻿ / - 22.73833 ; 31.00917 ﻿ ( Punda Maria Gate ) Pafuri Gate on the R525 road from Musina 22 ° 24 ′ 1 '' S 31 ° 2 ′ 29 '' E ﻿ / ﻿ 22.40028 ° S
31.04139 ° E ﻿ / - 22.40028 ; 31.04139 ﻿ ( Pafuri Gate ) Wilderness trails ( edit ) Nine different trails are on offer in the Kruger National Park . Some are
overnight and they last several days in areas of wilderness virtually untouched by humans . There are no set trails in the wilderness areas ; a visitor walks
along paths made by animals or seeks out new routes through the bush .
</item 2>

<item 3>
Kruger National Park
3 Red bush - willow and mopane veld 4.1. 4 Shrub mopane veld 4.2 Birds 4.3 Mammals 4.4 Reptiles 4.5 Amphibians and fish 5 Anti-poaching measures 5.1 Poachers
5.2 Rhino horn 5.3 Ivory stocks 5.4 Other 6 Accommodation and facilities 6.1 Rest camps 6.2 Designated private lodges 6.3 Bushveld camps 6.4 Restaurants 7 Gates
to the Kruger Park 8 Wilderness trails 9 Gallery 10 See also 11 References 12 External links History ( edit ) Main article : History of the Kruger National Park
Pre-reserve ( -- 1898 ) ( edit ) The area that the park covers today was part of the last wild frontier in the eastern half of Transvaal before the Second Anglo
- Boer War . Paul Kruger , President of the Republic of South Africa at the time , proclaimed the area , which was inhabited by the Tsonga people , a sanctuary
for the protection of its wildlife .
</item 3>

<item 4>
Kruger National Park
Kruger National Park - Wikipedia Kruger National Park Jump to : navigation , search Kruger National reserve IUCN category II ( national park ) Burchell 's zebra
in a Kruger landscape Location of the park ( red area ) in South Africa Location Limpopo and Mpumalanga provinces , South Africa Nearest city Nelspruit (
southern ) Phalaborwa ( central ) Coordinates 24 ° 0 ′ 41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Coordinates : 24 ° 0 ′
41 '' S 31 ° 29 ′ 7 '' E ﻿ / ﻿ 24.01139 ° S 31.48528 ° E ﻿ / - 24.01139 ; 31.48528 Area 19,485 km ( 7,523 sq mi ) Established 31 May 1926 ( 1898 ) Visitors
1,336,981 ( in 2004 ) Governing body South African National Parks www.sanparks.
</item 4>

<item 5>
Kruger National Park
It is one of the largest national parks in the world , with an area of 19,485 square kilometres ( 7,523 sq mi ) . The park is approximately 360 kilometres ( 220
mi ) long , and has an average width of 65 kilometres ( 40 mi ) . At its widest point , the park is 90 kilometres ( 56 mi ) wide from east to west . To the
north and south of the park two rivers , the Limpopo and the Crocodile respectively , act as its natural boundaries . To the east the Lebombo Mountains separate
it from Mozambique . Its western boundary runs parallel with this range , roughly 65 kilometres ( 40 mi ) distant . The park varies in altitude between 200
metres ( 660 ft ) in the east and 840 metres ( 2,760 ft ) in the south - west near Berg - en - Dal . The highest point in the park is here , a hill called
Khandzalive . Several rivers run through the park from west to east , including the Sabie , Olifants , Crocodile , Letaba , Luvuvhu and Limpopo Rivers .
</item 5>

<item 6>
Kruger National Park
1946 -- 1994 ( edit ) Plaque in the park . Now and then people do get killed , however , this is extremely rare . Stevenson - Hamilton was replaced by Colonel
J.A.B. Sandenbergh of the South African Air Force . During 1959 , work commenced to completely fence the park boundaries . Work started on the southern boundary
along the Crocodile River and in 1960 the western and northern boundaries were fenced , followed by the eastern boundary with Mozambique . The purpose of the
fence was to curb the spread of diseases , facilitate border patrolling and inhibit the movement of poachers . The Makuleke area in the northern part of the
park was forcibly taken from the Makuleke people by the government in 1969 and about 1500 of them were relocated to land to the south so that their original
tribal areas could be integrated into the greater Kruger National Park . 1994 -- present ( edit ) In 1996 the Makuleke tribe submitted a land claim for 19,842
hectares ( 198.42 km ) in the northern part of the Kruger National Park .
</item 6>

<item 7>
Kruger National Park
The land was given back to the Makuleke people , however , they chose not to resettle on the land but to engage with the private sector to invest in tourism ,
thus resulting in the building of several game lodges . In the late 1990s , the fences between the Kruger Park and Klaserie Game Reserve , Olifants Game Reserve
and Balule Game Reserve were dropped and incorporated into the Greater Kruger Park with 40 000 hectares added to the Reserve . In 2002 , Kruger National Park ,
Gonarezhou National Park in Zimbabwe , and Limpopo National Park in Mozambique were incorporated into a peace park , the Great Limpopo Transfrontier Park .
Location and geography ( edit ) Geography of the Kruger National Park Olifants River Luvuvhu and Limpopo rivers at Crookes Corner in Pafuri triangle The park
lies in the north - east of South Africa , in the eastern parts of Limpopo and Mpumalanga provinces . Phalaborwa , Limpopo is the only town in South Africa that
borders the Kruger National Park .
</item 7>

<item 8>
Kruger National Park
That motion , introduced together with another Volksraad member by the name of R.K. Loveday , and accepted for discussion in September 1895 by a majority of one
vote , resulted in the proclamation by Paul Kruger , president of the Transvaal Republic ( South African Republic ) , on 26 March 1898 , of a `` Government
Wildlife Park . '' This park would later be known as the Sabi Game Reserve . The park was initially created to control hunting and protect the diminished number
of animals in the park . James Stevenson - Hamilton became the first warden of the reserve in 1902 . The reserve was located in the southern one - third of the
modern park . Shingwedzi Reserve , named after the Shingwedzi River and now in northern Kruger National Park , was proclaimed in 1903 . During the following
decades all the native tribes were removed from the reserve and during the 1960s the last were removed at Makuleke in the Pafuri triangle . In 1926 , Sabie Game
Reserve , the adjacent Shingwedzi Game Reserve , and farms were combined to create Kruger National Park .
</item 8>

<item 9>
Kruger National Park
fin24.com/Companies/TravelAndLeisure/Kruger-Park-closes-as-eateries-move-in-20131216 External links ( edit ) Find more aboutKruger National Parkat Wikipedia 's
sister projects Media from Wikimedia Commons Travel guide from Wikivoyage Map all coordinates using : OpenStreetMap Google Maps Download coordinates as : KML
GPX Kruger National Park official website Webcams in the Kruger National parks of South Africa Addo Elephant Agulhas ǀAi - ǀAis / Richtersveld Augrabies
Bontebok Camdeboo Garden Route Golden Gate Highlands Karoo Kgalagadi Kruger Mapungubwe Marakele Mokala Mountain Zebra Namaqua Table Mountain Tankwa Karoo
Tsitsikamma West Coast VIAF : 151887382 ISNI : 0000 0001 2186 0681 GND : 4098727 - 9 Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Kruger_National_Park&oldid=837962811 '' Categories : IUCN Category II Protected areas established in 1926 Kruger
National Park National parks of South
</item 9>

<item 10>
Kruger National Park
and min . temperatures in ° F Precipitation totals in inches The climate of the Kruger National Park and Lowveld is subtropical . Summer days are humid and hot
. The rainy season is from September until May . The Kruger National Park website lists September and October as the driest periods , culminating in rains late
in October . The dry winter season is the ideal time to visit this region for various reasons . There is less chance of contracting malaria and the days are
milder . Viewing wildlife is more rewarding as the vegetation is more sparse and animals are drawn to the waterholes to drink every morning and evening . Flora
and Fauna ( edit ) Vegetation ( edit ) Plants life in the park consists of four main areas : Natural landscape for much of the Park Thorn trees and red bush -
willow veld ( edit ) This area lies between the western boundary and roughly the centre of the park south of the Olifants River .
</item 10>
=== END ITEM id=bfc8a7308e984ffd ===

=== ITEM id=11f782622e3d0d70 ===
Gold sentences and facts:
Sentence 0: Rank Team Won Lost Tied Pct . First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744 568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684 572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens 181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401 11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400 460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392 468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32 Tampa Bay Buccaneers 250 393 . 389 1976 644 NFC South
  Fact 0.0: The table columns are rank, team, won, lost, tied, pct, first NFL season, total games, and division.
  Fact 0.1: The Dallas Cowboys won 493 games, lost 367, tied 6, had a pct of .573, had a first NFL season of 1960, played 866 total games, and are in the NFC East.
  Fact 0.2: The Green Bay Packers won 730 games, lost 553, tied 37, had a pct of .567, had a first NFL season of 1921, played 1,320 total games, and are in the NFC North.
  Fact 0.3: The Chicago Bears won 744 games, lost 568, tied 42, had a pct of .565, had a first NFL season of 1920, played 1,354 total games, and are in the NFC North.
  Fact 0.4: The Miami Dolphins won 439 games, lost 341, had a pct of .563, had a first NFL season of 1966, played 784 total games, and are in the AFC East.
  Fact 0.5: Rank 5 is the New England Patriots, who won 476 games, lost 383, tied 9, had a pct of .554, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.6: Rank 6 is the New York Giants, who won 684 games, lost 572, tied 33, had a pct of .543, had a first NFL season of 1925, played 1,289 total games, and are in the NFC East.
  Fact 0.7: Rank 7 is the Denver Broncos, who won 465 games, lost 393, tied 10, had a pct of .541, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.8: Rank 8 is the Minnesota Vikings, who won 457 games, lost 387, tied 10, had a pct of .541, had a first NFL season of 1961, played 854 total games, and are in the NFC North.
  Fact 0.9: Rank 9 is the Baltimore Ravens, who won 181 games, lost 154, had a pct of .540, played 336 total games, and are in the AFC North.
  Fact 0.10: Rank 10 is the San Francisco 49ers, who won 522 games, lost 450, tied 14, had a pct of .537, had a first NFL season of 1950, played 986 total games, and are in the NFC West.
  Fact 0.11: Rank 11 is the Indianapolis Colts, who won 502 games, lost 441, tied 7, had a pct of .532, had a first NFL season of 1953, played 950 total games, and are in the AFC South.
  Fact 0.12: Rank 12 is the Oakland Raiders, who won 456 games, lost 401, tied 11, had a pct of .532, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.13: Rank 13 is the Pittsburgh Steelers, who won 601 games, lost 535, tied 20, had a pct of .529, had a first NFL season of 1933, played 1,156 total games, and are in the AFC North.
  Fact 0.14: Rank 14 is the Kansas City Chiefs, who won 447 games, lost 409, tied 12, had a pct of .522, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.15: Rank 15 is the Washington Redskins, who won 586 games, lost 572, tied 28, had a pct of .506, had a first NFL season of 1932, played 1,186 total games, and are in the NFC East.
  Fact 0.16: Rank 16 is the Seattle Seahawks, who won 325 games, lost 318, had a pct of .505, had a first NFL season of 1976, played 644 total games, and are in the NFC West.
  Fact 0.17: Rank 17 is the Cleveland Browns, who won 462 games, lost 466, tied 10, had a pct of .498, had a first NFL season of 1950, played 938 total games, and are in the AFC North.
  Fact 0.18: Rank 18 is the Los Angeles Chargers, who won 426 games, lost 431, tied 11, had a pct of .497, had a first NFL season of 1960, played 868 total games, and are in the AFC West.
  Fact 0.19: Rank 19 is the Los Angeles Rams, who won 544 games, lost 554, tied 21, had a pct of .496, had a first NFL season of 1937, played 1,119 total games, and are in the NFC West.
  Fact 0.20: Rank 20 is the Carolina Panthers, who won 172 games, lost 179, had a pct of .490, had a first NFL season of 1995, played 352 total games, and are in the NFC South.
  Fact 0.21: Rank 21 is the Philadelphia Eagles, who won 555 games, lost 591, tied 26, had a pct of .485, had a first NFL season of 1933, played 1,172 total games, and are in the NFC East.
  Fact 0.22: Rank 22 is the Tennessee Titans, who won 413 games, lost 449, tied 6, had a pct of .479, had a first NFL season of 1960, played 868 total games, and are in the AFC South.
  Fact 0.23: Rank 23 is the Buffalo Bills, who won 400 games, lost 460, tied 8, had a pct of .465, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.24: Rank 24 is the Detroit Lions, who won 544 games, lost 641, tied 32, had a pct of .460, had a first NFL season of 1930, played 1,217 total games, and are in the NFC North.
  Fact 0.25: Rank 25 is the Cincinnati Bengals, who won 344 games, lost 408, had a pct of .458, had a first NFL season of 1968, played 756 total games, and are in the AFC North.
  Fact 0.26: Rank 26 is the New York Jets, who won 392 games, lost 468, tied 8, had a pct of .456, had a first NFL season of 1960, played 868 total games, and are in the AFC East.
  Fact 0.27: Rank 27 is the New Orleans Saints, who won 338 games, lost 427, tied 5, had a pct of .442, had a first NFL season of 1967, played 770 total games, and are in the NFC South.
  Fact 0.28: Rank 28 is the Houston Texans, who won 106 games, lost 134, tied 0, had a pct of .442, had a first NFL season of 2002, played 240 total games, and are in the AFC South.
  Fact 0.29: Rank 29 is the Jacksonville Jaguars, who won 155 games, lost 197, tied 0, had a pct of .440, had a first NFL season of 1995, played 352 total games, and are in the AFC South.
  Fact 0.30: Rank 30 is the Atlanta Falcons, who won 341 games, lost 437, tied 6, had a pct of .439, had a first NFL season of 1966, played 784 total games, and are in the NFC South.
  Fact 0.31: Rank 31 is the Arizona Cardinals, who won 542 games, lost 732, tied 40, had a pct of .428, had a first NFL season of 1920, played 1,314 total games, and are in the NFC West.
  Fact 0.32: Rank 32 is the Tampa Bay Buccaneers, who won 250 games, lost 393, had a pct of .389, had a first NFL season of 1976, played 644 total games, and are in the NFC South.

Retrieved items:
<item 1>
NFL win–loss records
NFL win -- loss records - wikipedia NFL win -- loss records Jump to : navigation , search It has been suggested that NFL standings since AFL -- NFL merger be
merged into this article . ( Discuss ) Proposed since October 2017 . The following is a list of win -- loss records for each of the 32 active National Football
League ( NFL ) teams . Contents ( hide ) 1 Regular season 2 Playoffs 3 Notes 4 References Regular season ( edit ) George Halas ( right ) of the Chicago Bears ,
who have recorded the most regular season wins . The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their regular
season win -- loss record percentage , accurate as of the end of the 2016 NFL regular season . For teams that previously played in the All - America Football
Conference ( AAFC ) , their AAFC win - loss records are not included as the NFL does not officially count AAFC statistics despite the 1950 NFL -- AAFC merger .
However , for teams that played in the American Football League ( AFL ) , their AFL win - loss records are included and officially counted by the NFL .
</item 1>

<item 2>
NFL win–loss records
540 336 AFC North 10 San Francisco 49ers 522 450 14 . 537 1950 986 NFC West 11 Indianapolis Colts 502 441 7 . 532 1953 950 AFC South 12 Oakland Raiders 456 401
11 . 532 1960 868 AFC West 13 Pittsburgh Steelers 601 535 20 . 529 1933 1,156 AFC North 14 Kansas City Chiefs 447 409 12 . 522 1960 868 AFC West 15 Washington
Redskins 586 572 28 . 506 1932 1,186 NFC East 16 Seattle Seahawks 325 318 . 505 1976 644 NFC West 17 Cleveland Browns 462 466 10 . 498 1950 938 AFC North 18 Los
Angeles Chargers 426 431 11 . 497 1960 868 AFC West 19 Los Angeles Rams 544 554 21 . 496 1937 1,119 NFC West 20 Carolina Panthers 172 179 . 490 1995 352 NFC
South 21 Philadelphia Eagles 555 591 26 .
</item 2>

<item 3>
List of National Football League records (individual)
Detroit Lions , 1992 -- 2010 Most wins against a single opponent , regular season ( and post season ) , on the road , by a starting quarterback : 14 , Tom Brady
, vs. Buffalo Bills , 2001 -- 2017 Most consecutive wins against a single opponent , regular season ( and post season ) , on the road , by a starting
quarterback : 11 , Brett Favre , Green Bay Packers vs. Chicago Bears , 1994 -- Jan. 2 , 2005 Fewest wins in a regular season by a starting quarterback who won
the Super Bowl : 0 , Doug Williams , Washington Redskins 1987 Oldest starting quarterback to win a playoff game : Tom Brady ( Jacksonville in the 2017 AFC
Finals ; 40 years 171 days ) . QB losses ( edit ) Note : These records are not listed in NFL Record and Fact Book Most career losses , regular season , by a
starting quarterback : 123 , Vinny Testaverde , 1987 -- 2007 Most career losses , post-season , by a starting quarterback : 13 , Peyton Manning , 1999 -- 2016 .
</item 3>

<item 4>
NFL win–loss records
355 2002 AFC North 30 Detroit Lions 20 7 13 . 350 2016 NFC North 31 Kansas City Chiefs 26 9 17 . 346 2016 AFC West 32 Cincinnati Bengals 19 5 14 . 263 2015 AFC
North Notes ( edit ) Jump up ^ For example , the Browns ' regular season win -- loss record does not include a 47 -- 4 -- 3 record compiled in the All - America
Football Conference from 1946 to 1949 , nor does their playoff record include the 5 -- 0 record compiled during the same period . Likewise , the 49ers ' AAFC
regular season win -- loss record of 38 -- 14 -- 2 , nor their 1 - 1 AAFC playoff record are officially counted Jump up ^ The Patriots played as the `` Boston
Patriots '' from 1960 -- 1970 , before adopting the `` New England Patriots '' name , which they have played under since the 1971 season . The team was also
briefly known as the `` Bay State Patriots '' prior to the 1971 season .
</item 4>

<item 5>
NFL win–loss records
Ties are registered as a half win and a half loss when calculating the win - loss record . Best win -- loss record in division Rank Team Won Lost Tied Pct .
First NFL Season Total Games Division Dallas Cowboys 493 367 6 . 573 1960 866 NFC East Green Bay Packers 730 553 37 . 567 1921 1,320 NFC North Chicago Bears 744
568 42 . 565 1920 1,354 NFC North Miami Dolphins 439 341 . 563 1966 784 AFC East 5 New England Patriots 476 383 9 . 554 1960 868 AFC East 6 New York Giants 684
572 33 . 543 1925 1,289 NFC East 7 Denver Broncos 465 393 10 . 541 1960 868 AFC West 8 Minnesota Vikings 457 387 10 . 541 1961 854 NFC North 9 Baltimore Ravens
181 154 . 540 336 AFC North 10 San Francisco 49ers 522 450 14 .
</item 5>

<item 6>
List of National Football League records (individual)
Most career road losses , post-season , by a starting quarterback : 7 , Brett Favre , 1993 -- 2009 Most career home losses , post-season , by a starting
quarterback : 6 , Peyton Manning , 1999 -- 2016 . Most consecutive losses , post-season , by a starting quarterback : 4 , Y.A. Tittle , 1957 , 1961 -- 1963 ;
Warren Moon 1991 -- 1994 ; Andy Dalton 2011 -- 2014 Most consecutive road losses , post-season , by a starting quarterback : 6 , Dave Krieg , 1983 , 1984 , 1987
, 1992 -- 1994 Most consecutive road losses by a starting quarterback : 14 , Steve DeBerg , 1978 -- 1979 Most consecutive home losses by a starting quarterback
: 10 , Archie Manning , 1979 -- 1980 ; Chris Weinke , 2001 -- 2002 , 2006 Most road losses in a season , by a rookie starting quarterback : 8 , Peyton Manning ,
Indianapolis Colts 1998 ; Derek Carr , Oakland Raiders ,
</item 6>

<item 7>
NFL win–loss records
389 1976 644 NFC South Playoffs ( edit ) Members of the Baltimore Ravens franchise , the team with the highest playoff win -- loss percentage , during the
Lombardi Trophy presentation following their Super Bowl XLVII victory Marvin Lewis , head coach of the Cincinnati Bengals , who have registered the lowest
playoff win - loss percentage of all time The following is a listing of all 32 current National Football League ( NFL ) teams ranked by their playoff win - loss
record percentage , accurate as of the end of Super Bowl LI . Best win -- loss record in division Team active in playoffs Rank Team Games Won Lost Pct . Last
playoff appearance Division Baltimore Ravens 23 15 8 . 652 2014 AFC North New England Patriots 51 32 19 . 627 2016 AFC East Green Bay Packers 56 34 22 . 607
2016 NFC North 4 ( tie ) Pittsburgh Steelers 60 36 24 . 600 2016 AFC North 4 ( tie ) San Francisco 49ers 50 30 20 . 600 2013 NFC West 6 Oakland Raiders 44 25 19
.
</item 7>

<item 8>
NFL win–loss records
480 AFC East 18 Philadelphia Eagles 40 19 21 . 475 2013 NFC East 19 Jacksonville Jaguars 11 5 6 . 455 2007 AFC South 20 Los Angeles Rams 43 19 24 . 442 NFC West
21 ( tie ) Arizona Cardinals 16 7 9 . 438 2015 NFC West 21 ( tie ) New Orleans Saints 16 7 9 . 438 2013 NFC South 23 Houston Texans 7 . 429 2016 AFC South 24
Tennessee Titans 33 14 19 . 424 2008 AFC South 25 Atlanta Falcons 22 9 13 . 409 2016 NFC South 26 Minnesota Vikings 47 19 28 . 404 2015 NFC North 27 Tampa Bay
Buccaneers 15 6 9 . 400 2007 NFC South 28 Los Angeles Chargers 28 11 17 . 393 2013 AFC West 29 Cleveland Browns 31 11 20 . 355 2002 AFC North 30 Detroit Lions
20 7 13 .
</item 8>

<item 9>
List of National Football League records (individual)
Peyton Manning , Indianapolis Colts 1998 ; Derek Carr , Oakland Raiders , 2014 Most losses in a season , by a starting quarterback : 15 , Archie Manning , New
Orleans Saints , 1980 ; Jeff George , Indianapolis Colts , 1991 ; Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses , by a rookie starting
quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most losses in a season , by a rookie starting quarterback : 15 , Deshone Kizer , Cleveland Browns ,
2017 Most consecutive losses in a season , by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses to start a season ,
by a starting quarterback : 15 , Deshone Kizer , Cleveland Browns , 2017 Most consecutive losses by a starting quarterback : 21 , Dan Pastorini , Houston Oilers
, 1972 -- 1974 Most consecutive losses by a starting quarterback , regular season , to start a career : 15 , Deshone Kizer , Cleveland Browns , 2017 Most
consecutive road losses by a starting quarterback , regular season , to start a career : 13 ,
</item 9>

<item 10>
NFL win–loss records
490 1995 352 NFC South 21 Philadelphia Eagles 555 591 26 . 485 1933 1,172 NFC East 22 Tennessee Titans 413 449 6 . 479 1960 868 AFC South 23 Buffalo Bills 400
460 8 . 465 1960 868 AFC East 24 Detroit Lions 544 641 32 . 460 1930 1,217 NFC North 25 Cincinnati Bengals 344 408 . 458 1968 756 AFC North 26 New York Jets 392
468 8 . 456 1960 868 AFC East 27 New Orleans Saints 338 427 5 . 442 1967 770 NFC South 28 Houston Texans 106 134 0 . 442 2002 240 AFC South 29 Jacksonville
Jaguars 155 197 0 . 440 1995 352 AFC South 30 Atlanta Falcons 341 437 6 . 439 1966 784 NFC South 31 Arizona Cardinals 542 732 40 . 428 1920 1,314 NFC West 32
Tampa Bay Buccaneers 250 393 .
</item 10>
=== END ITEM id=11f782622e3d0d70 ===

=== ITEM id=614c928caaf420de ===
Gold sentences and facts:
Sentence 0: Character Actor Job House Song ( s ) Miss Hoolie Julie Wilson Nimmo Nursery teacher Green house Nursery Song : Everybody everyone Day off song : What do you want to do today ( Series 1 -- 2 ) Strike up the band ( Series 3 -- 4 ) Archie the Inventor Miles Jupp Inventor Pink castle I 'm Archie the Inventor ( Series 1 - 4 ) Great Inventions , Groovy Solutions ( Series 3 -- 4 ) Josie Jump Buki Akib ( series 1 - 2 ) Kasia Haddad ( Series 3 -- 4 ) Fitness instructor Yellow house Josie Jump is My Name ( Jump a Little Higher ) ( Series 1 - 4 ) Cheer You Up ( Series 3 -- 4 ) Edie McCredie Juliet Cadzow Bus driver Blue garage / house When I Honk My Horn ( Series 1 - 4 ) Let Me Take You on a Journey ( Series 3 -- 4 ) PC Plum Andrew Agnew Policeman Black and white police station I 'm PC Plum ( Series 1 -- 4 ) Follow the Clue ( Series 3 -- 4 ) Spencer the Painter Rodd Christensen Painter / Musician Orange house Climbing Up My Musical Ladder ( Series 1 -- 4 ) If You Need a Little Rhythm ( Series 3 -- 4 ) Suzie Sweet Mary Riggans Shop and café workers Red shop I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Suzie 's Cooking ( Series 3 -- 4 ) Penny Pocket Kim Tserkezie I 'm Suzie Sweet , I 'm Penny Pocket ( Series 1 -- 4 ) Sort It ( Series 3 -- 4 )
  Fact 0.0: Miss Hoolie is played by Julie Wilson Nimmo.
  Fact 0.1: Miss Hoolie's job is nursery teacher.
  Fact 0.2: Miss Hoolie's house is the Green house (Nursery).
  Fact 0.3: Miss Hoolie's Nursery Song is Everybody everyone.
  Fact 0.4: Miss Hoolie's Day off song is What do you want to do today (Series 1-2).
  Fact 0.5: Miss Hoolie's Day off song is Strike up the band (Series 3-4).
  Fact 0.6: Archie the Inventor is played by Miles Jupp.
  Fact 0.7: Archie the Inventor's job is inventor.
  Fact 0.8: Archie the Inventor's house is the Pink castle.
  Fact 0.9: Archie the Inventor's song is I'm Archie the Inventor (Series 1-4).
  Fact 0.10: Archie the Inventor's song is Great Inventions, Groovy Solutions (Series 3-4).
  Fact 0.11: Josie Jump is played by Buki Akib in Series 1-2.
  Fact 0.12: Josie Jump is played by Kasia Haddad in Series 3-4.
  Fact 0.13: Josie Jump's job is fitness instructor.
  Fact 0.14: Josie Jump's house is the Yellow house.
  Fact 0.15: Josie Jump's song is Josie Jump is My Name (Jump a Little Higher) (Series 1-4).
  Fact 0.16: Josie Jump's song is Cheer You Up (Series 3-4).
  Fact 0.17: Edie McCredie is played by Juliet Cadzow.
  Fact 0.18: Edie McCredie's job is bus driver.
  Fact 0.19: Edie McCredie's house is the Blue garage / house.
  Fact 0.20: Edie McCredie's song is When I Honk My Horn (Series 1-4).
  Fact 0.21: Edie McCredie's song is Let Me Take You on a Journey (Series 3-4).
  Fact 0.22: PC Plum is played by Andrew Agnew.
  Fact 0.23: PC Plum's job is policeman.
  Fact 0.24: PC Plum's house is the Black and white police station.
  Fact 0.25: PC Plum's song is I'm PC Plum (Series 1-4).
  Fact 0.26: PC Plum's song is Follow the Clue (Series 3-4).
  Fact 0.27: Spencer the Painter is played by Rodd Christensen.
  Fact 0.28: Spencer the Painter's job is painter and musician.
  Fact 0.29: Spencer the Painter's house is the Orange house.
  Fact 0.30: Spencer the Painter's song is Climbing Up My Musical Ladder (Series 1-4).
  Fact 0.31: Spencer the Painter's song is If You Need a Little Rhythm (Series 3-4).
  Fact 0.32: Suzie Sweet is played by Mary Riggans.
  Fact 0.33: Suzie Sweet's job is shop and café worker.
  Fact 0.34: Suzie Sweet's house is the Red shop.
  Fact 0.35: Suzie Sweet's song is I'm Suzie Sweet, I'm Penny Pocket (Series 1-4).
  Fact 0.36: Suzie Sweet's song is Suzie's Cooking (Series 3-4).
  Fact 0.37: Penny Pocket is played by Kim Tserkezie.
  Fact 0.38: Penny Pocket's song is I'm Suzie Sweet, I'm Penny Pocket (Series 1-4).
  Fact 0.39: Penny Pocket's song is Sort It (Series 3-4).

Retrieved items:
<item 1>
Balamory
This episode also featured Greg Hemphill ( Julie Wilson Nimmo 's husband in real life ) and John Altman , who plays Nick Cotton in the BBC soap opera EastEnders
. Celebrity chef Keith Floyd appeared in Suzie Sweet 's song `` Suzie 's Cookin ' '' . Occasionally , the shots of children entering the nursery are cut down
for repeats for time reasons . Retirement ( edit ) On 30 April 2005 , the BBC announced that it would not order another season of the series , citing they
wanted to experiment and create more new programme ideas . In that year , all the cast ( except for two of them ) performed a farewell touring show called ``
Balamory Live : Strike up the Band '' , a follow up to the `` Balamory Live : What 's The Story '' tour from 2004 . By the time the programme was retired , many
of the cast were also keen to move on . Miles Jupp , who played Archie , stated that his role caused some problems when he was performing stand up on the
Edinburgh Fringe , due to parents assuming his show was child friendly , and bringing their children to see it .
</item 1>

<item 2>
Balamory
Edie McCredie drives the nursery school bus . She has travelled all over the world , and has a variety of photographs and souvenirs to talk about . She is also
an expert driver and motor mechanic , and occasionally gets exasperated with other road users . Police officer PC Plum aspires to be a master detective , but
Balamory is a crime free village . He spends most of his time watching wildlife and using his detective skills , to figure out what the animals have been doing
. He is also particularly interested in rules , regulations , and safety . Spencer is officially responsible for painting the coloured houses of Balamory , but
also paints pictures and creates music . His painter 's ladder is musical , each rung sounding a different note . Spencer is also the only American to reside in
Balamory and he is black . Suzie Sweet and Penny Pocket together run the village shop and cafe . Suzie Sweet is an older , grandmotherly figure who loves to
tell customers about the items in the shop . Penny Pocket is a young woman who uses a wheelchair ; she has a talent for mathematics .
</item 2>

<item 3>
Balamory is a Scottish preschool series created by Brian Jameson and produced by BBC Scotland, airing from 10 February 2002 to 14 April 2005. Its 254 episodes
follow Miss Hoolie and a colourful island community as characters solve simple problems through visits, recaps and songs. Filmed mainly in Tobermory on Mull,
the show featured eight central characters, including an inventor, fitness instructor, bus driver, policeman and wheelchair-using shopkeeper. It won the
</item 3>

<item 4>
Balamory
She is advised to visit PC Plum , who advises her to feed it vegetables . She then goes to the shop run by Pocket and Sweet to buy rabbit food , and finally
brings the rabbit to Miss Hoolie 's nursery for the children to feed and pat it . Each episode of Balamory contains at least five songs : `` What 's the Story
in Balamory ? '' , the theme song Miss Hoolie 's opening song , either `` Everybody , Everyone '' if the nursery school is open , or one of the two `` day off
'' songs otherwise `` Which Coloured House Are We Going To ? '' , sung by the main character after presenting his or her problem Character - specific songs ,
sung by the characters visited by the main character Episode guide ( edit ) Series One Series Two Series Three Series Four Audience ( edit ) The programme was
aimed at pre schoolers . The programme 's creator had intended it as a `` soap opera for children '' . The show can also be seen in a humorous light . The
characters in Balamory are somewhat naive and light hearted , often appearing to be children trapped in adult bodies with adult roles .
</item 4>

<item 5>
Balamory
Balamory - wikipedia Balamory Jump to : navigation , search This article needs additional citations for verification . Please help improve this article by
adding citations to reliable sources . Unsourced material may be challenged and removed . ( December 2010 ) ( Learn how and when to remove this template message
) Balamory Created by Brian Jameson Starring Julie Wilson Nimmo Miles Jupp Kasia Haddad Buki Akib Juliet Cadzow Andrew Agnew Rodd Christensen Mary Riggans Kim
Tserkezie Country of origin United Kingdom Original language ( s ) English No. of episodes 254 , including Christmas special . Production Producer ( s ) Brian
Jameson Running time 19 -- 21 minutes . 28 -- 30 minutes .
</item 5>

<item 6>
Balamory
The storyline revolves around simple problems designed to challenge pre schoolers ' minds . Songs ( edit ) There are many songs in Balamory , mainly sung by the
actors . These songs are used consistently throughout the series , when corresponding characters or circumstances are introduced . In the run starting from
September 2004 , new songs were introduced . These songs are longer and take up more time in the episode , so the old versions were still often used . In
addition to the character specific songs mentioned in the `` Characters '' section , there are two general songs : `` What 's the Story in Balamory ? '' is the
show 's theme song , sung over the opening titles and played as an instrumental over the closing credits . `` Which Coloured House Are We Going To ? '' is sung
by the main character of each episode after introducing that episode 's problem . Production ( edit ) Very few episodes feature actors other than the main
characters and the nursery children . Terry Wogan made a guest appearance in one episode ( The Game Show ) as a television director .
</item 6>

<item 7>
Balamory
Each character also has one or more individual songs . Character Actor Job House Song ( s ) Miss Hoolie Julie Wilson Nimmo Nursery teacher Green house Nursery
Song : Everybody everyone Day off song : What do you want to do today ( Series 1 -- 2 ) Strike up the band ( Series 3 -- 4 ) Archie the Inventor Miles Jupp
Inventor Pink castle I 'm Archie the Inventor ( Series 1 - 4 ) Great Inventions ,
</item 7>

<item 8>
Balamory
Penny Pocket is a young woman who uses a wheelchair ; she has a talent for mathematics . Episodes ( edit ) A typical episode follows this general plan : The
opening credits show the town of `` Balamory '' with its brightly coloured houses . Miss Hoolie opens the nursery school , greets the nursery children , and
talks about that day 's weather . One of the other characters ( who will be the main character of that episode ) tells Miss Hoolie about a problem . This
character then sings `` Which Coloured House Are We Going To ? '' , asking the young viewers to guess which of the Balamory characters will be most helpful in
solving the problem . The main character then proceeds to visit several other characters until the problem is solved . Miss Hoolie and the main character
provide a summary of the story at each point , and a final recap at the end of the episode . For example , in one episode Josie Jump has been asked to look
after a neighbour 's rabbit , but does not know what it needs . She is advised to visit PC Plum , who advises her to feed it vegetables .
</item 8>

<item 9>
Balamory
It was also shown on TLC . In Ireland , episodes were shown until recently on weekdays on RTÉ Two 's The Den and continue to be shown on The Den 's replacement
, RTÉjr . Tobermory , Mull , the setting for the fictional town of Balamory Edie McCredie 's bus ( LDV Convoy ) in the Glasgow Museum of Transport Contents (
hide ) 1 Characters 2 Episodes 2.1 Episode guide 3 Audience 4 Songs 5 Production 6 Retirement 7 Awards 8 References 9 External links Characters ( edit ) There
are eight central characters in Balamory . Each character wears a distinctive colour of clothing like pink or yellow , and lives in a house of the same colour
like Archie lives in a pink castle , with the exception of Penny Pocket ( who shares the red house / shop with Suzie Sweet , and wears blue ) and PC Plum ( who
lives in the white house but wears a police uniform ) . Each character also has one or more individual songs .
</item 9>

<item 10>
Balamory
28 -- 30 minutes . Specials Production company ( s ) BBC Scotland Distributor BBC Release Original network CBeebies Picture format 16 : 9 Widescreen Audio
format Stereo Original release 10 February 2002 -- 14 April 2005 External links Website www.bbc.co.uk/cbeebies/shows/balamory Balamory is a Scottish live action
television series on British television ( BBC One , BBC Two and CBeebies ) for pre school children , about a fictional small island community called Balamory in
Scotland . It was produced between 10 February 2002 and 14 April 2005 by BBC Scotland , with 254 episodes made ( including a DVD exclusive Christmas episode ) .
The series was created by Brian Jameson . Balamory was filmed mostly in Tobermory on the Isle of Mull , with the exception of scenes at Archie 's castle filmed
at Fenton Tower in North Berwick , and other scenes , such as Camden and the nursery filmed in Glasgow . A select series of episodes was shown for two brief
seasons in the United States on the Discovery Kids channel . It was also shown on TLC .
</item 10>
=== END ITEM id=614c928caaf420de ===

=== ITEM id=128049bde6f78392 ===
Gold sentences and facts:
Sentence 0: In 1980 , The Shining became the basis for a film adaptation directed by Stanley Kubrick . Kubrick 's vision for the movie differed from King 's significantly in many ways , including the portrayal of the Overlook Hotel . The exteriors of Kubrick 's Overlook were supplied by the Timberline Lodge on the slopes of Mt . Hood in Oregon . Inspiration for the interior sets ( erected at Elstree Studios in England ) came from the 1927 Ahwahnee Hotel in Yosemite National Park . The management of the Timberline Lodge , fearful that guests would refuse to stay in their Room 217 if it were featured in a horror movie , insisted that Kubrick change the haunted room in the film to Room 237 .
  Fact 0.0: The Shining became the basis for a film adaptation in 1980.
  Fact 0.1: The Shining film adaptation was directed by Stanley Kubrick.
  Fact 0.2: Kubrick's vision for the movie differed from Stephen King's significantly in many ways.
  Fact 0.3: Kubrick's vision differed from King's in the portrayal of the Overlook Hotel.
  Fact 0.4: The exteriors of Kubrick's Overlook were supplied by the Timberline Lodge.
  Fact 0.5: The Timberline Lodge is on the slopes of Mt. Hood in Oregon.
  Fact 0.6: Inspiration for the interior sets of the Kubrick film came from the 1927 Ahwahnee Hotel in Yosemite National Park.
  Fact 0.7: The interior sets were erected at Elstree Studios in England.
  Fact 0.8: The management of the Timberline Lodge was fearful that guests would refuse to stay in their Room 217 if it were featured in a horror movie.
  Fact 0.9: The management of the Timberline Lodge insisted that Kubrick change the haunted room in the film to Room 237.

Retrieved items:
<item 1>
The Stanley Hotel
- Roe Emery Maurice L. Albertson , Civil Engineer at CSU Maxwell Abbell , Chicago real estate investor 1995 -- present - Grand Heritage Hotel Group See also (
edit ) Freelan Oscar Stanley The Shining ( novel ) The Shining ( film ) The Shining ( miniseries ) Colonial Revival Architecture Estes Park , CO National
Register of Historic Places listings in Larimer County , Colorado References ( edit ) ^ Jump up to : National Park Service ( 2008 - 04 - 15 ) . `` National
Register Information System '' . National Register of Historic Places . National Park Service . ^ Jump up to : `` Rocky Mountain Legends '' .
LegendsOfAmerica.com . Missing or empty url = ( help ) ; access - date = requires url = ( help ) ^ Jump up to : `` Rocky Mountain National Park - Culture '' .
US-Parks.com . Retrieved 2007 - 05 - 10 . Jump up ^ Turkewitz , Julie ( 3 September 2015 ) . `` Hotel That Inspired ' The Shining ' Builds on Its Eerie Appeal
'' . New York Times . Retrieved 17 September 2015 .
</item 1>

<item 2>
The Stanley Hotel
That night , according to King , `` I dreamed of my three - year - old son running through the corridors , looking back over his shoulder , eyes wide ,
screaming . He was being chased by a fire - hose . I woke up with a tremendous jerk , sweating all over , within an inch of falling out of bed . I got up , lit
a cigarette , sat in a chair looking out the window at the Rockies , and by the time the cigarette was done , I had the bones of The Shining firmly set in my
mind . '' The Shining was published in 1977 and became the third great success of King 's career after Carrie and ' Salem 's Lot . The Overlook Hotel - the
fictional hotel in King 's book inspired by the Stanley - is an evil entity haunted by its many victims . Room 217 of the Overlook features prominently in the
novel , having been the room where King spent the night at the Stanley . This is the room on the second floor in the center of the west wing with a balcony
overlooking the south terrace . Room 217 remains the hotel 's most requested accommodation .
</item 2>

<item 3>
The Stanley Hotel
The Hotel has also been used as a filming location for other movies and TV shows ; most notably , as the fictional `` Hotel Danbury '' of Aspen , CO , in the
1994 film Dumb and Dumber . From 2013 to 2015 , the hotel property hosted the Stanley Film Festival , an independent horror film festival operated by the Denver
Film Society , held in early May . The festival featured screenings , panels , student competitions , audience awards and receptions . The Stanley Film Festival
was put on hiatus in 2016 , and canceled for 2017 . The historic Stanley Concert Hall serves as venue for various musical groups such as country - punk band
Murder By Death which has performed a Shining - themed series of concerts in the space . Haunted reputation ( edit ) Despite a peaceful early history , in the
years following the publication of The Shining , The Stanley Hotel has gained a reputation among paranormal investigators for frequent activity . The hotel
offers guided `` Ghost Tours '' to guests and visitors which feature spaces reputed to be exceptionally active . The hotel has also served as a location for
numerous paranormal investigation shows such as Ghost Hunters and Ghost Adventures .
</item 3>

<item 4>
The Stanley Hotel
Famous guests ( edit ) The Stanley Hotel has hosted the following persons of note : 1934 - Erich Fromm , the German psychoanalyst 1936 - Gov. Alf Landon ( R ,
Kansas ) : opponent to Franklin Delano Roosevelt 's re-election 1974 - Stephen King , author who was inspired to write The Shining from his stay in room 217
1976 - Bob Dylan & Joan Baez , classic rock icons during the Rolling Thunder Revue tour 1994 - HM Emperor Akihito of Japan with consort and crown prince , on a
state visit to the US Ownership History ( edit ) 1908 - 26 - Freelan Oscar Stanley 1926 - 29 - The Stanley Corporation 1929 - 30 - Freelan Oscar Stanley 1930 -
?
</item 4>

<item 5>
The Stanley Hotel
Approximately five miles from the entrance to Rocky Mountain National Park , the Stanley offers panoramic views of Lake Estes , the Rockies and especially Long
's Peak . It was built by Freelan Oscar Stanley of Stanley Steamer fame and opened on July 4 , 1909 , catering to the American upper class at the turn of the
century . The hotel and its surrounding structures are listed on the National Register of Historic Places . The Stanley Hotel hosted the horror novelist Stephen
King , serving as inspiration for the Overlook Hotel in his 1977 bestseller The Shining and its 1980 film adaption of the same name , as well as the location
for the 1997 miniseries . Today , it includes a restaurant , spa , and bed - and - breakfast and provides guided tours which feature the history and alleged
paranormal activity of the site .
</item 5>

<item 6>
The Stanley Hotel
Contents ( hide ) 1 History 2 Gallery 3 Architecture 3.1 Main Building 3.2 Concert Hall 3.3 The Lodge 3.4 The Hedge Maze 4 The Shining 5 Film location and arts
venue 6 Haunted reputation 7 Famous Guests 8 Ownership History 9 See also 10 References 11 External links History ( edit ) In 1903 , the Yankee steam - powered
car inventor Freelan Oscar Stanley ( 1849 - 1940 ) was stricken with a life - threatening resurgence of tuberculosis . The most highly recommended treatment of
the day was fresh , dry air with lots of sunlight and a hearty diet . Therefore , like many `` lungers '' of his day , Stanley resolved to take the curative air
of Rocky Mountain Colorado . He and Flora arrived in Denver in March and , in June , decided to spend the rest of the summer in the mountains , in Estes Park .
Over the course of the season , Stanley 's health improved dramatically . Impressed by the beauty of the valley and grateful for his recovery , he decided to
return every year .
</item 6>

<item 7>
The Stanley Hotel
The hedge maze ( edit ) In 2015 , the open area in front of the hotel , originally a driveway for Stanley Steamers and a promenade for guests to enjoy mountain
views , was replaced with a hedge maze . This non-historic feature was added to evoke the hotel 's connection to The Shining . Although a hedge maze features
prominently in Stanley Kubrick 's version of The Shining , no such feature can be found in Stephen King 's novel . The lawn of the Overlook Hotel , as King
imagined it , was adorned with topiary animals and it was set that way for the shooting of the miniseries in 1996 . Historically , neither feature existed at
the Stanley . The design was the result of a competition , chosen from amongst 300 entrants from across the globe . The winning submission was designed by New
York architect Mairim Dallaryan Standing . In 2016 , John Cullen , owner of the Stanley Hotel , announced a competition for a sculpture to be the centerpiece of
the terrace in front of the hotel . Sculptors Sutton Betti and Daniel Glanz won the competition with a sculpture of F.O.
</item 7>

<item 8>
The Stanley Hotel
Water was supplied by the Black Canyon Creek which was dammed in 1906 . The style of the campus is so - called Colonial Revival . Although rare in the western
United States , F.O. Stanley chose the Colonial Revival for its fashionable popularity in New England where he had already designed his own home and a social
club in the style . The hotel 's clientele would presumably , like the Stanleys , have identified the style with New England respectability and sophistication
in contrast to the rusticity of the surrounding town . At one time , Stanley planned to build another , more economical hotel in Estes Park as well as a
headquarters and residence for the superintendent of the Rocky Mountain National Park in the same style , to harmonize with his grand hotel . While the forms
and layouts of the buildings are suited to their modern uses , their ornamentation exhibits the stylistic hallmarks of Georgian or Federal architecture from the
staunch symmetry of the south elevation to the cupola , dormers , Palladian windows , side - and fan - lights , scroll brackets and `` swan 's neck '' pediments
that articulate the exterior .
</item 8>

<item 9>
The Stanley Hotel
Room 217 remains the hotel 's most requested accommodation . Film location and arts venue ( edit ) In 1980 , The Shining became the basis for a film adaptation
directed by Stanley Kubrick . Kubrick 's vision for the movie differed from King 's significantly in many ways , including the portrayal of the Overlook Hotel .
The exteriors of Kubrick 's Overlook were supplied by the Timberline Lodge on the slopes of Mt . Hood in Oregon . Inspiration for the interior sets ( erected at
Elstree Studios in England ) came from the 1927 Ahwahnee Hotel in Yosemite National Park . The management of the Timberline Lodge , fearful that guests would
refuse to stay in their Room 217 if it were featured in a horror movie , insisted that Kubrick change the haunted room in the film to Room 237 . In 1997 The
Shining TV series was produced , with the Stanley Hotel as the primary shooting location . A playhouse version of the main building which adorned the lawn of
the Overlook Hotel in the series can be viewed in the basement of the Stanley .
</item 9>

<item 10>
The Stanley Hotel
Stanley holding one of his violins . The sculpture was installed and dedicated on September 29 , 2016 . They used another hotel to film the outside shots and
the Stanley for inside shots . The Shining ( edit ) Cover of The Shining , first edition , 1977 On October 30 , 1974 , horror writer Stephen King and his wife
Tabitha spent one night at the Stanley Hotel during their one year of residency in Boulder , Colorado . They were driving through Estes Park as night approached
when they happened upon the Stanley Hotel . They decided to book a room and , upon checking in , discovered that they were the only overnight guests . `` They
were just getting ready to close for the season , and we found ourselves the only guests in the place -- with all those long , empty corridors . '' He and his
wife were served dinner in an empty dining room accompanied by canned orchestral music . `` Except for our table all the chairs were up on the tables . So the
music is echoing down the hall , and , I mean , it was like God had put me there to hear that and see those things . ''
</item 10>
=== END ITEM id=128049bde6f78392 ===
