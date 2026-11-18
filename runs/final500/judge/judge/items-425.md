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

This file has 10 items.

=== ITEM id=3322101f9df9289c ===
Gold sentences and facts:
Sentence 0: The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US skyscraper.
  Fact 0.0: The Woolworth Building is at 233 Broadway, Manhattan, New York City.
  Fact 0.1: The Woolworth Building was designed by architect Cass Gilbert.
  Fact 0.2: The Woolworth Building was constructed between 1910 and 1912.
  Fact 0.3: The Woolworth Building is an early US skyscraper.
Sentence 1: 1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.
  Fact 1.0: 1 New York Plaza is an office building in New York City's Financial District.
  Fact 1.1: 1 New York Plaza was built in 1969.
  Fact 1.2: 1 New York Plaza is at the intersection of South and Whitehall Streets.

Retrieved items:
<item 1>
90 West Street
90 West Street (alternatively West Street Building) is a building in Lower Manhattan, New York City.

 It was designed by architect Cass Gilbert and structural engineer Gunvald Aus for the West Street Improvement Corporation.

 When completed in 1907, the building's Gothic styling and ornamentation served to emphasize its 23-story height, and foreshadowed Gilbert's later work on the
Woolworth Building.

 Originally built as an office building, the main tenant was the Delaware, Lackawanna and Western Railroad and the top floor was occupied by Garret's
Restaurant, which advertised itself as the "world's highest restaurant".
</item 1>

<item 2>
F. W. Woolworth Building (Watertown, New York)
The Woolworth Building is an historic building in Watertown, New York.

 It is a contributing building in the Public Square Historic District.

 Plans for the Woolworth Building were begun in 1916 by Frank W. Woolworth, the founder of the Woolworth's chain of department stores.
</item 2>

<item 3>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 3>

<item 4>
Woolworth Building
The Woolworth Building, at 233 Broadway, Manhattan, New York City, designed by architect Cass Gilbert and constructed between 1910 and 1912, is an early US
skyscraper.

 The original site for the building was purchased by F. W. Woolworth and his real estate agent Edward J. Hogan by April 15, 1910, from the Trenor Luther Park
Estate and other owners for $1.65 million.

 By January 18, 1911, Woolworth and Hogan had acquired the final site for the project, totaling $4.5 million.

 More than a century after its construction, it remains, at 241.4 m , one of the 100 tallest buildings in the United States as well as one of the 30 tallest
buildings in New York City.

 It has been a National Historic Landmark since 1966, and a New York City landmark since 1983.
</item 4>

<item 5>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 5>

<item 6>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 6>

<item 7>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 7>

<item 8>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 8>

<item 9>
The entries cover notable U.S. buildings, districts, businesses, and people. New York features prominently: office towers such as 888 Seventh Avenue, 245 Park
Avenue, 383 and 270 Madison/Park Avenue, and Goldman Sachs’ 200 West Street; landmarks including the Woolworth Building and 90 West Street; and residential or
mixed-use towers such as MiMA and Random House Tower. Other topics include historic theaters, retail and media companies, transportation firms, and real-estate
developer
</item 9>

<item 10>
Ware's Department Store
Ware’s Department Store is a historic building located in the city of New Rochelle in Westchester County, New York and is significant for both architectural and
commercial reasons.

 Ware's was Westchester’s first and, for many years, largest department store, and was prominently located on New Rochelle’s fashionable Main Street.

 The store operated from 1881 to the late 1930s, when the property was sold to the retailer Bloomingdale's to serve as their first suburban department store
location.

 Not only was Ware's a key place in New Rochelle during its 20th-century boom years, its founder and owner, Howard R. Ware was a leading figure in the rapidly
growing community as well.

 Ware first moved to New Rochelle from Massachusetts at the age of 13 and began to work as a clerk.

 In 1881 he became partner in the firm of Ware & Sheffield, which eventually became a stock company in 1913.

 renamed He was a director and vice president of the National City Bank of New Rochelle, a founder and first president of the local Y.M.C.A. from 1899 to 1916,
and an active member of St.
</item 10>
=== END ITEM id=3322101f9df9289c ===

=== ITEM id=3243a7631d3b5a2b ===
Gold sentences and facts:
Sentence 0: Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
  Fact 0.0: Hänt Extra is a Swedish language magazine.
  Fact 0.1: Hänt Extra is a weekly magazine.
  Fact 0.2: Hänt Extra is a celebrity and youth magazine.
  Fact 0.3: Hänt Extra is published by Aller Media.
  Fact 0.4: Hänt Extra is published in Stockholm, Sweden.
Sentence 1: Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
  Fact 1.0: Tea Moderna is a Macedonian woman's weekly magazine.
  Fact 1.1: Tea Moderna's name in Macedonian is Теа Модерна.
  Fact 1.2: Tea Moderna is a magazine for women.

Retrieved items:
<item 1>
Makedonsko Sonce
Makedonsko Sonce (Macedonian: ; English: Macedonian Sun ) is a weekly magazine published in the Republic of Macedonia.

 The title means "Macedonian Sun", referring to the Vergina Sun, which is used in the logo of the magazine.

 It was established by Gjorgija (George) Atanasoski and the first edition was published on June 24, 1994.

 It describes its own political stance as affirming "Macedonian national values, not only on territory of Macedonia but also in the other parts of ethnic
Macedonia".
</item 1>

<item 2>
2015 Macedonian Football Supercup
The 2015 Macedonian Supercup was the 3rd Macedonian Football Supercup, an annual Macedonian football match played between the winners of the previous season's
First League and Macedonian Cup.

 The game will be played between Rabotnički, who beat Teteks to win the 2015 Macedonian Cup Final, and Vardar, champions of the 2014–15 First League.

 It was played at Philip II Arena, Skopje and was won by Vardar by 4–3 on penalties after the match was ended 1–1 after the regular and extra time.
</item 2>

<item 3>
Albanian Coalition &quot;Perspective&quot;
The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
</item 3>

<item 4>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 4>

<item 5>
Katarina Ivanovska
Katarina Ivanovska (Macedonian: Катарина Ивановска ; born 18 August 1988) is a Macedonian model and actress.

 She began her modeling career in 2004, appearing at Milan Fashion Week after winning the Look Models International model search in Macedonia.

 In December, 2004, she appeared in a pictorial for "Elle" magazine and has also appeared in "Citizen K", "Stiletto" and the Italian and Russian "Vogue".

 She has been featured on the covers of "Diva" and "Maxima" magazines and in advertisements for D&G in 2006.

 She is considered the most successful Macedonian model.

 In 2010, Ivanovska appeared in Serbian "Elle" magazine.

 In 2011 she signed a contract for advertising Victoria's Secret products.

 In 2011 she got her first acting job in the Macedonian World War II film, "The Third Half", landing the lead role of a young Jewish girl named Rebecca.
</item 5>

<item 6>
Nikola Mladenov
Nikola Mladenov (Macedonian: Никола Младенов ; 10 March 1964 – 26 March 2013) was a Macedonian journalist, actor, founder and managing editor of the weekly
newspaper "Fokus", widely regarded as one of the most reputable journalists in Macedonia, who along with his colleagues from the newspaper "Mlad borec" is
considered to have laid the basis for the pluralistic principles in the Macedonian journalism.

 The "Award for best research story of the year – Nikola Mladenov", which is awarded by the Macedonian Institute for Media, is named after him.
</item 6>

<item 7>
Tea Moderna (magazine)
Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
</item 7>

<item 8>
Den Za Nas
Den Za Nas (in Macedonian: Ден за нас ) is the title of the fourth Macedonian language album by the Macedonian superstar Toše Proeski released in 2004.

 It is a 2-CD set which contains all of the songs performed on the National final in order to select a song for the Eurovision Song Contest 2004 with which
Proeski would represent Macedonia in the contest.

 The winning song was "Life" which qualified for the final and finished on the 14th place in the final.

 Some extra songs such as "Dzvezdo Severnice", "Me Sakaš So Zborovi", "Ima Li Den Za Nas" and others were recorded especially for the album.
</item 8>

<item 9>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 9>

<item 10>
Football Weekly
Football Weekly is a podcast about football, hosted by "The Guardian" newspaper in the United Kingdom.

 Originally provided weekly (as its name suggests), its popularity led to a mid-weekly spinoff, "Football Weekly Extra".

 "Football Weekly" airs on Mondays during the European football season, after the weekend's fixtures, while "Football Weekly Extra" airs on Thursdays.

 During the Euro 2008, World Cup 2010, Euro 2012, World Cup 2014 and Euro 2016 tournaments the podcast aired daily.
</item 10>
=== END ITEM id=3243a7631d3b5a2b ===

=== ITEM id=fd6803e21c03d1f6 ===
Gold sentences and facts:
Sentence 0:  Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.0: Reed is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.1: Mike Pringle is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.2: Johnny Bright is one of the players most often mentioned as being the greatest running back in CFL history.
Sentence 1: Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.
  Fact 1.0: Johnny D. Bright was born June 11, 1930.
  Fact 1.1: Johnny D. Bright died December 14, 1983.
  Fact 1.2: Johnny D. Bright was a professional Canadian football player.
  Fact 1.3: Johnny D. Bright played in the Canadian Football League.
Sentence 2:  He played college football at Drake University.
  Fact 2.0: Johnny D. Bright played college football at Drake University.

Retrieved items:
<item 1>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 1>

<item 2>
Skip Peete
Skip Peete (born January 30, 1963) is an American football coach who serves as the current running backs coach for the Los Angeles Rams of the National Football
League (NFL).

 Prior to the Rams, he coached running backs for the Chicago Bears, Dallas Cowboys, and Oakland Raiders.

 He also has experience in college, coaching running backs for two years at UCLA, wide receivers at Michigan State and Rutgers, and both running backs and wide
receivers at the University of Pittsburgh.
</item 2>

<item 3>
Madre Hill
Madre Hill (born January 2, 1976) is a former American football running back, playing last for the Oakland Raiders of the National Football League.

 Considered one of the greatest running backs to come out of the University of Arkansas , Hill was named 1st Team All-SEC in 1995 and was named to the
Razorbacks' All-time team for the 1990s.

 He formerly held the all-time season rushing record for Arkansas high schools and for the Arkansas Razorbacks.
</item 3>

<item 4>
Harry Prowell
Harry Prowell A.A.(10 July 1936 – 27 June 2000) was a Guyanese long distance runner who represented Guyana in the Marathon at the 1968 Summer Olympics in Mexico
City, Mexico.

 He is known to be one of the greatest Marathon runners Guyana has ever produced, setting the national record in 1968.

 To date, he is the only Guyanese ever to compete in the Marathon at the Summer Olympic Games and one of the most prominent Indo-Caribbean long distance runners
of his time.

 He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth
Games in Edinburgh, Scotland.
</item 4>

<item 5>
George Reed (Canadian football)
George Robert Reed, CM, SOM (born October 2, 1939), is a former American college football and Canadian Football League player.

 Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.

 In November 2006, Reed was voted one of the CFL's Top 50 players (#2) of the league's modern era by Canadian sports network TSN.
</item 5>

<item 6>
Damon Allen
Damon Allen (born July 29, 1963) is a former professional quarterback who played in the Canadian Football League.

 He is currently second in all-time professional football passing yards and second in all-time CFL passing yards after he was surpassed for first place by the
Montréal Alouettes' Anthony Calvillo on October 10, 2011.

 He is pro-football's combined passing & rushing leader with a total of 84,301 yards.

 Allen retired as professional football's all-time leading passer with 72,381 passing yards after he surpassed Warren Moon's total of 70,553 yards (in both the
CFL and NFL combined) on September 4, 2006 in the annual Labour Day Classic.

 He also retired in third place in all-time CFL rushing yards with 11,920 yards, behind Mike Pringle and George Reed.

 The 2007 season marked Allen's twenty-third season in the CFL and he officially announced his retirement on May 28, 2008 at age 44.

 Allen is the younger brother of Pro Football Hall of Famer Marcus Allen.
</item 6>

<item 7>
2,000-yard club
The 2,000-yard club is a group of seven National Football League (NFL) running backs that have rushed for 2,000 or more yards in a season.

 These seven rushing seasons rank as the highest single-season rushing totals in NFL history, and reaching the 2,000-yard mark is considered a significant
achievement for running backs.

 No running back has yet achieved this feat twice.

 The first 2,000-yard season was recorded in 1973 by Buffalo Bills running back O.J. Simpson.

 He is the only player to have surpassed 2,000 yards in a 14-game season, as all others occurred in 16-game seasons; he finished the season with 2,003 rushing
yards, averaging six yards per carry and an NFL-record 143.1 rushing yards per game.

 Los Angeles Rams running back Eric Dickerson, who had broken the single-season rookie rushing record in 1983, recorded the second 2,000-yard season in 1984.

 Dickerson rushed for 2,105 yards, the current NFL rushing record, and averaged 131.6 rushing yards per game.
</item 7>

<item 8>
Marc Mueller
Marc Mueller (born April 15, 1989) is the current running backs coach of the Calgary Stampeders of the Canadian Football League (CFL).

 In college, he played quarterback for the Regina Rams of the CIS.

 He signed as an undrafted free agent with the Edmonton Eskimos of the CFL on May 9, 2011 following the 2011 CFL Draft.

 He was released on June 20, 2011 in order to play his final year for the Rams.

 Mueller is the grandson of former CFL quarterback and coach Ron Lancaster.
</item 8>

<item 9>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 9>

<item 10>
Dolph Pulliam
Adolphus "Dolph" Pulliam (born March 14, 1946 in West Point, Mississippi) is a former American basketball player.

 He played collegiately at Drake University.

 He was named to Drake University's All-Century basketball team.

 He helped lead the Drake men's basketball team to the 1969 Final Four and a third-place finish.

 On 7 February 2009, Drake University retired Pulliam's #5 jersey that he wore for the Bulldogs.
</item 10>
=== END ITEM id=fd6803e21c03d1f6 ===

=== ITEM id=1bc24dde779c6eaf ===
Gold sentences and facts:
Sentence 0: German Spitz is used to refer to both a breed of dog and category or type of dog.
  Fact 0.0: German Spitz is used to refer to a breed of dog.
  Fact 0.1: German Spitz is used to refer to a category or type of dog.
Sentence 1:  Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.
  Fact 1.0: Several modern breeds have been developed from the German Spitz.
  Fact 1.1: These modern breeds are either registered as separate breeds.
  Fact 1.2: These modern breeds are either registered as varieties of German Spitz.
Sentence 2:  The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that would later become known as the English Springer Spaniel.
  Fact 2.0: The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels.
  Fact 2.1: Attempts were made to use the term to specify a breed that would later become known as the English Springer Spaniel.

Retrieved items:
<item 1>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 1>

<item 2>
King Charles Spaniel
The King Charles Spaniel (also known as the English Toy Spaniel) is a small dog breed of the spaniel type.

 In 1903, the Kennel Club combined four separate toy spaniel breeds under this single title.

 The other varieties merged into this breed were the Blenheim, Ruby and Prince Charles Spaniels, each of which contributed one of the four colours available in
the breed.
</item 2>

<item 3>
American Eskimo: German Spitz-derived companion/watchdog, renamed amid WWI prejudice; popular U.S. circus performer. English Springer: affectionate gun dog for
flushing/retrieving, also sniffer; recognized 1902, with show/working lines. German Spitz Klein: small German Spitz-type toy/utility dog. Picardy Spaniel: older
French hunting spaniel, weatherproof; now less popular, prone to ear infections. Welsh Springer: red-and-white hunting dog, loyal
</item 3>

<item 4>
American Cocker Spaniel
The American Cocker Spaniel is a breed of sporting dog.

 It is a spaniel type dog that is closely related to the English Cocker Spaniel; the two breeds diverged during the 20th century due to differing breed
standards in America and the UK.

 In the United States, the breed is usually called the Cocker Spaniel, while elsewhere in the world, it is called the American Cocker Spaniel in order to
differentiate between it and its English cousin, which was already known as "Cocker Spaniel" before the American variety was created.

 The word "cocker" is commonly held to stem from their use to hunt woodcock in England, while "spaniel" is thought to be derived from the type's origins in
Spain.
</item 4>

<item 5>
English Springer Spaniel
The English Springer Spaniel is a breed of gun dog in the Spaniel family traditionally used for flushing and retrieving game.

 It is an affectionate, excitable breed with a typical lifespan of twelve to fourteen years.

 They are very similar to the Welsh Springer Spaniel and are descended from the Norfolk or Shropshire Spaniels of the mid-19th century; the breed has diverged
into separate show and working lines.

 The breed suffers from average health complaints.

 The show-bred version of the breed has been linked to "rage syndrome", although the disorder is very rare.

 It is closely related to the Welsh Springer Spaniel and very closely to the English Cocker Spaniel; less than a century ago, springers and cockers would come
from the same litter.

 The smaller "cockers" hunted woodcock while the larger littermates were used to flush, or "spring," game.

 In 1902, The Kennel Club recognized the English Springer Spaniel as a distinct breed.

 They are used as sniffer dogs on a widespread basis.

 The term "Springer" comes from the historic hunting role, where the dog would flush (spring) birds into the air.
</item 5>

<item 6>
Norfolk Spaniel
The Norfolk Spaniel or Shropshire Spaniel is an extinct breed of dog since the early 20th century.

 It was originally thought to have originated from the work of one of the Dukes of Norfolk, but this theory was disproven after being in doubt during the later
part of the 19th century.

 The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that
would later become known as the English Springer Spaniel.
</item 6>

<item 7>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 7>

<item 8>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 8>

<item 9>
Welsh Springer Spaniel
The Welsh Springer Spaniel is a breed of dog and a member of the spaniel family.

 Thought to be comparable to the old Land Spaniel, they are similar to the English Springer Spaniel and historically have been referred to as both the Welsh
Spaniel and the Welsh Cocker Spaniel.

 They were relatively unknown until a succession of victories in dog trials by the breed increased its popularity.

 Following recognition by The Kennel Club in 1902, the breed gained the modern name of Welsh Springer Spaniel.

 The breed's coat only comes in a single colour combination of white with red markings, usually in a piebald pattern.

 Loyal and affectionate, they can become very attached to family members and are wary of strangers.

 Health conditions are limited to those common among many breeds of dog, although they are affected more than average by hip dysplasia and some eye conditions.

 They are a working dog, bred for hunting, and while not as rare as some varieties of spaniel, they are rarer than the more widely known English Springer
Spaniel with which they are sometimes confused.
</item 9>

<item 10>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 10>
=== END ITEM id=1bc24dde779c6eaf ===

=== ITEM id=353e60bca46bbb35 ===
Gold sentences and facts:
Sentence 0:  The Park is home to the first flume ride in Ireland (Pirates Plunge).
  Fact 0.0: The Park is home to the first flume ride in Ireland.
  Fact 0.1: The first flume ride in Ireland is named Pirates Plunge.
Sentence 1: Grace O'Malley (c. 1530 – c. 1603; also Gráinne O'Malley, Irish: "Gráinne Ní Mháille" ) was lord of the Ó Máille dynasty in the west of Ireland, following in the footsteps of her father Eoghan Dubhdara Ó Máille.
  Fact 1.0: Grace O'Malley lived from about 1530 to about 1603.
  Fact 1.1: Grace O'Malley is also called Gráinne O'Malley.
  Fact 1.2: Grace O'Malley's Irish name is Gráinne Ní Mháille.
  Fact 1.3: Grace O'Malley was lord of the Ó Máille dynasty in the west of Ireland.
  Fact 1.4: Grace O'Malley followed in the footsteps of her father, Eoghan Dubhdara Ó Máille.

Retrieved items:
<item 1>
Tanglewood Music and Arts Festival
Tanglewood Music and Arts Festival is an annual two-day music, comedy and arts event held within the grounds of Narrow Water Castle, County Down, Northern
Ireland.

 The festival, founded in 2011, has attracted significant attention both within the local area and around Northern Ireland with buses coming from as far as
Belfast to the event.

 For the 2011 and 2012 festivals Tanglewood was held on one day; however for 2013 it has expanded into a two-day camping festival.
</item 1>

<item 2>
White Water Landing (Dorney Park)
White Water Landing is a Shoot-the-Chutes ride at Dorney Park & Wildwater Kingdom amusement park in Allentown, Pennsylvania.

 It was built in 1992, 10 years after a sister park named Cedar Point Built a defunct flume using the same name: White Water Landing (Cedar Point).

 This ride is identical to Snake River Falls at Cedar Point in Sandusky, Ohio.

 It is located in the upper side near Hydra the Revenge.
</item 2>

<item 3>
Glenwood Caverns Adventure Park
Glenwood Caverns Adventure Park is an adventure park located above Glenwood Springs, Colorado, about 160 miles west of Denver.

 Prior to 2003, only cave tours were available until a major expansion took place.

 The park is unique because it sits at an altitude of 7,100 feet on a mountain above Glenwood.

 Today, the park features numerous attractions in addition to the cave tours.
</item 3>

<item 4>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 4>

<item 5>
Wild West Express Coaster
Wild West Express Coaster is a steel roller coaster operating at Glenwood Caverns Adventure Park in Glenwood Springs, Colorado.

 Wild West Express Coaster opened to the public at Glenwood Caverns Adventure Park on May 25, 2012.
</item 5>

<item 6>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 6>

<item 7>
Deering Oaks
Deering Oaks is a 55 acre public park in Portland, Maine, which has a baseball diamond, tennis courts, a playground, and a pond.

 It is located west of downtown Portland and is bordered by Deering Avenue on the west, Forest Avenue on the east, Park Avenue to the south and Interstate 295
to the north.

 State Street bisects the park.

 Access is from State Street, Deering Avenue, or Park Avenue.

 The Portland Farmers Market is located on the Park Avenue side of the park.

 The park is listed on the National Register of Historic Places.
</item 7>

<item 8>
Christchurch Adventure Park
Christchurch Adventure Park is an Adventure park in the Port Hills of Christchurch, New Zealand.

 Built by the Canadian company Select Evolution, it had its opening function on 16 December 2016.

 The 1500 holders of special passes could ride from 17 December, while the park opened to the public on 21 December.

 According to the developer, the 358 ha park with a 1.8 km chairlift and initially 50 km of downhill tracks is the largest facility of its kind in the Southern
Hemisphere.

 The park closed on 13 February as a precaution due to a nearby fire and two days later, most of the park's tree cover had been destroyed in a large wildfire.

 The lift and ziplines have suffered significant damage, and the park's condition is much worse than the operator initially thought.

 No reopening date has been set, and "it won't be a quick fix".
</item 8>

<item 9>
Luna Park, Schenectady
Roughly seven decades before the Skycoaster rides that now dot various United States amusement parks, Luna/Rexford Park featured an aerial swing ride.
</item 9>

<item 10>
Pirate Adventure Park
Pirate Adventure Park is a small theme park located beside Westport House in Westport, Ireland.

 The Park is home to the first flume ride in Ireland (Pirates Plunge).

 The Park is Pirate themed due to Wesport House's association with Grace O'Malley.

 The Park was developed by The 11th Marquess of Sligo on the grounds of Westport House.
</item 10>
=== END ITEM id=353e60bca46bbb35 ===

=== ITEM id=8f934c69c93ccf5a ===
Gold sentences and facts:
Sentence 0: "Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".
  Fact 0.0: 'Into Dust' is a song by the American duo Mazzy Star.
  Fact 0.1: 'Into Dust' appears as the ninth track on the second album of Mazzy Star.
  Fact 0.2: The second album of Mazzy Star is 'So Tonight That I Might See'.
Sentence 1: Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and bassist Kendra Smith.
  Fact 1.0: Mazzy Star is an American alternative rock band.
  Fact 1.1: Mazzy Star was formed in Santa Monica, California, in 1989.
  Fact 1.2: Mazzy Star was formed from the group Opal.
  Fact 1.3: Opal was a collaboration of guitarist David Roback and bassist Kendra Smith.

Retrieved items:
<item 1>
Assembly of Dust
Assembly of Dust (also known as AOD) is an American rock band formed in 2002 by former Strangefolk frontman Reid Genauer.

 After the breakup of Strangefolk, Genauer decided to record a solo album, and he recruited some friends from Strangefolk's days on the road to help out.

 After titling the 2003 solo release “Assembly of Dust,” Genauer decided to use the name for his new group.

 AOD originally featured Genauer on lead vocals and guitar, Nate Wilson on the keyboard, Adam Terrell on lead guitar, John Leccese on bass, and Andy Herrick on
drums.

 In recent years the band has experienced some transformations, and the current status of the band now features Reid Genauer on lead vocals and guitar, Adam
Terrell on lead guitar, John Leccese on bass, Jason Crosby on the keys and violin, and Dave Diamond on the drums.

 Despite the changes in the band's lineup, AOD has managed to maintain the same high energy and tight musical finesse for which they are known.
</item 1>

<item 2>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 2>

<item 3>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 3>

<item 4>
Mazzy Star
Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and
bassist Kendra Smith.

 Roback's friend Hope Sandoval became the group's vocalist when Smith left the band.
</item 4>

<item 5>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 5>

<item 6>
Into Dust
"Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".

 The song's arrangement is sparse and features only acoustic guitar and violin as accompaniment to the vocals.
</item 6>

<item 7>
Candlelight Red
Candlelight Red is a rock band from Williamsport, Pennsylvania.

 They have produced two studio albums and an EP.

 Their EP "Demons" and album "Reclamation" was produced by Morgan Rose of Sevendust.
</item 7>

<item 8>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 8>

<item 9>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 9>

<item 10>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 10>
=== END ITEM id=8f934c69c93ccf5a ===

=== ITEM id=9bfa33e874080aad ===
Gold sentences and facts:
Sentence 0: Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist Tayeb Salih.
  Fact 0.0: Season of Migration to the North is a classic post-colonial Sudanese novel.
  Fact 0.1: Season of Migration to the North is written by the novelist Tayeb Salih.
  Fact 0.2: The Arabic title of Season of Migration to the North is موسم الهجرة إلى الشمال, transliterated Mawsim al-Hiǧra ilā ash-Shamāl.
Sentence 1:  The novel is a counter narrative to "Heart of Darkness".
  Fact 1.0: Season of Migration to the North is a counter narrative to Heart of Darkness.
Sentence 2: Heart of Darkness (1899) is a novella by Polish-British novelist Joseph Conrad, about a voyage up the Congo River into the Congo Free State, in the heart of Africa, by the story's narrator Charles Marlow.
  Fact 2.0: Heart of Darkness is dated 1899.
  Fact 2.1: Heart of Darkness is a novella.
  Fact 2.2: Heart of Darkness is by Polish-British novelist Joseph Conrad.
  Fact 2.3: Heart of Darkness is about a voyage up the Congo River into the Congo Free State.
  Fact 2.4: The voyage in Heart of Darkness goes into the heart of Africa.
  Fact 2.5: The narrator of Heart of Darkness is Charles Marlow.

Retrieved items:
<item 1>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 1>

<item 2>
Season of Migration to the North
Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist
Tayeb Salih.

 Originally published in Arabic in 1966, it has since been translated into more than twenty languages.

 Salih was fluent in both English and Arabic, but chose to pen this novel in Arabic.

 The English translation was published in 1969 as part of the influential Heinemann African Writers Series.

 The novel is a counter narrative to "Heart of Darkness".

 It was described by Edward Said as one of the six great novels in Arabic literature.

 In 2001 it was selected by a panel of Arab writers and critics as the most important Arab novel of the twentieth century.
</item 2>

<item 3>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 3>

<item 4>
Graph query row:
novelist: Joseph Conrad
occupation: None
connection: None
related_work: Season of Migration to the North
</item 4>

<item 5>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 5>

<item 6>
The Stranger (novel)
L’Étranger (The Outsider [UK], or The Stranger [US]) is a 1942 novel by French author Albert Camus.

 Its theme and outlook are often cited as examples of Camus' philosophy of the absurd and existentialism, though Camus personally rejected the latter label.
</item 6>

<item 7>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 7>

<item 8>
First North Americans
The novels take into account new developments in North American archaeology such as the discovery of Kennewick Man and the development of the coastal-route
model as a possible alternative or supplement to overland migration across Beringia.
</item 8>

<item 9>
First North Americans
First North Americans are a series of historical fiction novels published by Tor and written by husband and wife co-authors W. Michael Gear & Kathleen O'Neal
Gear.

 The series, which began with 1990's "People of the Wolf", explores various civilizations and cultures in prehistoric North America.

 It is comparable to Jean M. Auel's "Earth's Children" series, set in prehistoric Europe, but each of its books focuses on a different time period, location,
and set of characters.

 The first four novels form a coherent, more or less linear narrative, from the initial migration of Siberian peoples into what is now Canada and Alaska (People
of the Wolf) through the florescence of the Mississippian semi-urban mound-building culture, considered the "high-water mark" of North American pre-Columbian
civilization, around 1000 AD.

 The remaining novels cover a wide variety of times and settings, in no particular order, ranging from tropical Florida in the 6th millennium BC to the Chaco
Empire of the Southwest in the 13th century AD.
</item 9>

<item 10>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 10>
=== END ITEM id=9bfa33e874080aad ===

=== ITEM id=aa41832e657d41b7 ===
Gold sentences and facts:
Sentence 0: Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.
  Fact 0.0: Heavenly Creatures is a 1994 New Zealand psychological drama.
  Fact 0.1: Heavenly Creatures was directed by Peter Jackson.
  Fact 0.2: Peter Jackson co-wrote the screenplay with his partner, Fran Walsh.
  Fact 0.3: Heavenly Creatures is about the notorious 1954 Parker-Hulme murder case.
  Fact 0.4: The Parker-Hulme murder case took place in Christchurch, New Zealand.
Sentence 1:  The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.
  Fact 1.0: The main premise of Heavenly Creatures deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme.
  Fact 1.1: Pauline Parker and Juliet Hulme murder Pauline Parker's mother.
Sentence 2: Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk series.
  Fact 2.0: Anne Perry was born on 28 October 1938 as Juliet Marion Hulme.
  Fact 2.1: Anne Perry is an English author of historical detective fiction.
  Fact 2.2: Anne Perry is best known for her Thomas Pitt and William Monk series.

Retrieved items:
<item 1>
Parker–Hulme murder case
The Parker–Hulme murder case began in the city of Christchurch, New Zealand, on 22 June 1954, when Honora Rieper (also known as Honora Parker, her legal name)
was killed by her teenaged daughter, Pauline Parker, and Pauline's close friend Juliet Hulme (later known as Anne Perry).

 Parker was 16 at the time, while Hulme was 15.
</item 1>

<item 2>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 2>

<item 3>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 3>

<item 4>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 4>

<item 5>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 5>

<item 6>
Peter Elliott (actor)
Peter Elliott is a leading New Zealand actor.

 He has appeared in numerous television shows including "Shortland Street", "Gloss" and .

 He has also appeared in several movies including "Heavenly Creatures".

 Peter has a daughter Lucy Elliott who is an actress, playing character Dayna Jenkins on "Shortland Street" in 2013.
</item 6>

<item 7>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 7>

<item 8>
Heavenly Creatures
Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the
notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.

 The film features Melanie Lynskey and Kate Winslet in their screen debuts with supporting roles by Sarah Peirse, Diana Kent, Clive Merrison, and Simon
O'Connor.

 The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.

 The events of the film cover the period from their meeting in 1952 to the murder in 1954.
</item 8>

<item 9>
Anne Perry
Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk
series.

 At the age of fifteen she was convicted of participating in the murder of her friend's mother, in 1954.

 She changed her name after serving her five-year sentence.
</item 9>

<item 10>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 10>
=== END ITEM id=aa41832e657d41b7 ===

=== ITEM id=16fbf3117f4fb06b ===
Gold sentences and facts:
Sentence 0: Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
  Fact 0.0: Scott Treleaven is a Canadian artist.
  Fact 0.1: Scott Treleaven's work employs a variety of media.
  Fact 0.2: The media employed in Scott Treleaven's work include collage, film, video, drawing, photography and installation.
Sentence 1: Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.
  Fact 1.0: Edmund Mortimer's dates are 21 August 1874 to 21 May 1944.
  Fact 1.1: Edmund Mortimer was an American actor.
  Fact 1.2: Edmund Mortimer was an American film director.

Retrieved items:
<item 1>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 1>

<item 2>
Tommy Atkins (director)
Tommy Atkins was an American director of the silent and early sound film eras.

 Born on July 18, 1887 in Springfield, Massachusetts, he would make his entrance into the film industry as the assistant director to Ralph Ince on the 1920
silent film, "Out of the Snows".

 It would be another eight years before he would make another film, again as assistant director, this time for FBO Pictures, on another silent film, "Crooks
Can't Win".

 He'd work as the assistant director on another sixteen films between 1928 and 1934, the most notable of which would be 1933's "Morning Glory", directed by
Lowell Sherman and starring Katharine Hepburn and Douglas Fairbanks Jr..

 In 1934 he would be given the chance to helm his first picture, "The Silver Streak", which was one of the top money-makers for RKO Pictures that year.

 He would only direct two more films, the second of which, "Hi, Gaucho!

", he would also write the story for.
</item 2>

<item 3>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 3>

<item 4>
Richard Estes
Richard Estes (born May 14, 1932 in Kewanee, Illinois) is an American artist, best known for his photorealist paintings.

 The paintings generally consist of reflective, clean, and inanimate city and geometric landscapes.

 He is regarded as one of the founders of the international photo-realist movement of the late 1960s, with such painters as John Baeder, Ralph Goings, Chuck
Close, Audrey Flack, and Duane Hanson.

 Author Graham Thompson writes "One demonstration of the way photography became assimilated into the art world is the success of photorealist painting in the
late 1960s and early 1970s.

 It is also called super-realism or hyper-realism and painters like Richard Estes, Denis Peterson, Audrey Flack, and Chuck Close often worked from photographic
stills to create paintings that appeared to be photographs."
</item 4>

<item 5>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 5>

<item 6>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 6>

<item 7>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 7>

<item 8>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 8>

<item 9>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 9>

<item 10>
Henry Murdoch
Henry Murdoch (17 September, 1920 - 24 April, 1987), born as George Henry Murdock, was an Australian aboriginal actor and stockman who appeared in Australian
films of the 1940s and 1950s.

 He was working as stockman in Rockhampton when discovered by Ralph Smart, who was helping make "The Overlanders" (1946).

 The film's director, Harry Watt, later claimed Murdoch and fellow aboriginal actor Clyde Combo "proved to be first-class actors and were exceedingly quick
witted and intelligent.

 They certainly disproved the conventional idea that the Australian aboriginal is an animalistic caveman."
</item 10>
=== END ITEM id=16fbf3117f4fb06b ===

=== ITEM id=fb5c202b7139f34b ===
Gold sentences and facts:
Sentence 0: The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.
  Fact 0.0: The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.
  Fact 0.1: Alberta is a Canadian province.
  Fact 0.2: Alberta is governed under fundamental rules described in the Constitution of Alberta.
Sentence 1:  As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.
  Fact 1.0: Alberta's constitution is an unwritten constitution.
  Fact 1.1: It is typical of all Canadian provinces for their constitutions to be unwritten.
  Fact 1.2: It is typical of Westminster systems more generally for their constitutions to be unwritten.
Sentence 2: The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.
  Fact 2.0: The Westminster system is a parliamentary system of government.
  Fact 2.1: The Westminster system is modelled after the system of government that developed in the United Kingdom.

Retrieved items:
<item 1>
The material surveys legislatures, institutions, and historical sites. The West Indies Federation’s territories became nine independent states and four British
territories; Guyana and Belize were observers. Hawaii, Alaska, Maryland, and Quebec have bicameral or historically bicameral legislatures; Alaska’s is the
smallest bicameral U.S. legislature. Oregon’s territorial legislature preceded statehood. Hawaii’s Capitol houses state government. Quebec’s legislature became
unicameral in 1968; Majorca’s Gran i General
</item 1>

<item 2>
Canada grew from British North American colonies into a federation in 1867; Newfoundland and Labrador joined in 1949. The Statute of Westminster (1931) granted
legislative sovereignty, completed by the Canada Act (1982). The Stamp Act taxed colonial printed materials, prompting the 1765 Stamp Act Congress. The West
Indies Federation (1958–62) collapsed amid political disputes. Other topics include the Royal Mint, archaeological hoards, Canadian file sharing, Wood Green’s
war memorial, Québec
</item 2>

<item 3>
Governance of England
There has not been a government of England since 1707 when the Kingdom of England ceased to exist as a sovereign state, as it merged with the Kingdom of
Scotland to form the Kingdom of Great Britain.

 Kingdom of Great Britain continued from 1707 until 1801 when it merged with the Kingdom of Ireland to form the United Kingdom of Great Britain and Ireland,
which itself became the United Kingdom of Great Britain and Northern Ireland (UK) in 1922 (in reality; in name in 1927) upon independence for most of the island
of Ireland.

 The UK since then has gone through significant change to its system of government, with devolved parliaments, assemblies and governments in Scotland, Wales and
Northern Ireland.

 England, however, remains under the full jurisdiction, on all matters, of the Parliament of the United Kingdom of Great Britain and Northern Ireland and the UK
government as no devolved administration has been created for England within the new structure.

 This situation has led to the anomaly, known as the West Lothian question, which is that Scottish Members of Parliament (MPs) are able to vote on legislation
that affects only England whereas English MPs can not vote on certain Scottish matters due to devolution.
</item 3>

<item 4>
Westminster system
The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.

 This term comes from the Palace of Westminster, the seat of the British parliament.
</item 4>

<item 5>
Quebec Legislature
The Quebec Legislature (officially Parliament of Quebec, French: "Parlement du Québec" ) is the legislature of the province of Quebec, Canada, since
Confederation in 1867.

 Today, the legislature is made of two elements: the monarch of Canada, represented by the Lieutenant Governor of Quebec, and the unicameral assembly called the
National Assembly of Quebec.

 The legislature has existed since 1867 when Quebec, then called Canada East, became one of the founding colonies of Canadian Confederation.

 From 1867 until 1968 the legislature was bicameral, containing a lower chamber called the Legislative Assembly of Quebec and an upper chamber called the
Legislative Council of Quebec.
</item 5>

<item 6>
Politics of Canada
The politics of Canada function within a framework of parliamentary democracy and a federal system of parliamentary government with strong democratic
traditions.

 Canada is a constitutional monarchy, in which the Monarch is head of state.

 The country has a multi-party system in which many of its legislative practices derive from the unwritten conventions of and precedents set by the United
Kingdom's Westminster Parliament.

 However, Canada has evolved variations: party discipline in Canada is stronger than in the United Kingdom and more parliamentary votes are considered motions
of confidence, which tends to diminish the role of non-Cabinet Members of Parliament, (MPs).

 Such members, in the government caucus, and junior or lower-profile members of opposition caucuses, are known as backbenchers. Backbenchers can, however, exert
their influence by sitting in parliamentary committees, like the Public Accounts Committee or the National-Defence Committee.
</item 6>

<item 7>
Constitution of Alberta
The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.

 As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.

 Alberta's constitution, like Britain's (on which it is modeled), includes any and all pieces of legislation, court decisions, proclamations, and conventions
which together inform how the province operates.

 Many statutes are important to understanding the governance of the province, but nowhere are they consolidated into a single document or even a list.

 The office of Attorney-General at one time suggested 23 acts which might be included, but cautioned that this was not a "definitive list".

 However, since Alberta is a part of federation, its powers are clearly delineated in law, via the Constitution of Canada.
</item 7>

<item 8>
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 8>

<item 9>
British North America Acts
The British North America Acts 1867–1975 are the original names of a series of Acts at the core of the constitution of Canada.

 They were enacted by the Parliament of the United Kingdom and the Parliament of Canada.

 In Canada, some of the Acts were amended or repealed by the Constitution Act, 1982.

 The rest were renamed in Canada as the "Constitution Acts".

 In the United Kingdom, those Acts that were passed by the British Parliament remain under their original names.

 The term "British North America" (BNA) refers to the British colonies in North America.
</item 9>

<item 10>
Canada under British rule
Canada first came under British rule with the Treaty of Paris (1763) which ceded New France, of which Canada was a part, to the British Empire.

 Gradually, other territories, colonies, and provinces that were part of British North America would be added to Canada.

 The Royal Proclamation of 1763 enlarged the colony of Canada under the name of the Province of Quebec, which with the Constitutional Act 1791 became known as
The Canadas.

 With the Act of Union 1840 Upper and Lower Canada were joined to become the United Province of Canada.

 Later, with Confederation in 1867, the British maritime colonies of New Brunswick and Nova Scotia were joined with the British colony of Canada to form the
Dominion of Canada, which was subsequently divided into four provinces, Ontario, Quebec, New Brunswick, and Nova Scotia.

 A number of other British colonies, such as Newfoundland and British Columbia, and large territories such as Rupert's Land initially remained outside of the
newly formed federation.

 Over time, the remaining colonies and territories within British North America came under the control of Canada until the current geographic extent of the
country was reached when Newfoundland and Labrador joined Canada in 1949.
</item 10>
=== END ITEM id=fb5c202b7139f34b ===
