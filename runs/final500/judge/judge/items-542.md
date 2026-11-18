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

This file has 11 items.

=== ITEM id=d85ff0fdae24096f ===
Gold sentences and facts:
Sentence 0:  The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
  Fact 0.0: The area along Northern Boulevard is well known for its high-end premium open-air shopping center.
  Fact 0.1: The high-end premium open-air shopping center is named the Americana Manhasset.
Sentence 1: International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine, Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van Sweden.
  Fact 1.0: Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine, Tiffany & Co., and Prada are international fashion designer labels and luxury goods companies.
  Fact 1.1: These companies maintain a total of more than 60 retail stores and restaurants.
  Fact 1.2: These stores and restaurants are on a 220000 sqft site.
  Fact 1.3: The 220000 sqft site is surrounded by gardens.
  Fact 1.4: The gardens surrounding the site were designed by Oheme van Sweden.

Retrieved items:
<item 1>
Christian Dior SE
Christian Dior SE (] ), commonly known as Dior, is a French luxury goods company controlled and chaired by French businessman Bernard Arnault, who also heads
LVMH – the world's largest luxury group.

 Dior itself holds 42.36% shares of and 59.01% voting rights within LVMH.
</item 1>

<item 2>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 2>

<item 3>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 3>

<item 4>
Miracle Mile (Manhasset)
The Miracle Mile is a prominent shopping district in Manhasset on the North Shore of Long Island in Nassau County, New York.

 The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
</item 4>

<item 5>
Concord watch
Concord Watch Company is a Swiss luxury goods company that is part of the Movado group that owns Movado, Ebel, ESQ, Coach and Hugo Boss.

 Founded in 1908, Concord was purchased in 1970 by the North American Watch Company, which also distributed the Piaget and Corum lines of watches.

 From the late 1970s to the late 1980s, Concord produced what would become some of the most marketed and respected luxury quartz watches on the market.

 Concord watches became recognized in large cities as status symbols and were noted for their innovation and design.

 Flagship quartz models such as the Concord Centurion and Concord Delirium ranged from $2,000 to $20,000 surpassing the price of base automatic Rolex, Cartier
and Omega wristwatches.

 By the 1990s Concord watches fell out of style and grew obsolete amidst the re-branding of the company.

 While the brand still exists today, it never regained its market position nor visibility that it once had.
</item 5>

<item 6>
Mouawad
Mouawad is a privately held Swiss and Emirati luxury goods company that makes fashion accessories, jewelry and retail company that sources diamonds and
gemstones, designs, manufacturers, and sells jewelry collections, objects of art, and luxury watches.

 The firm has headquarters in Geneva, Switzerland, with a Middle East headquarters at Jumeirah Lakes Towers in Dubai.

 Founded in 1891 in Beirut, Lebanon by David Mouawad, the firm is now led by 4th generation co-guardians Fred Mouawad, Alain Mouawad and Pascal Mouawad.

 The firm's jewelry and watch design and manufacturing divisions are located in Switzerland and Asia Pacific.
</item 6>

<item 7>
Americana Manhasset
International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine,
Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van
Sweden.

 Peter Marino is Americana Manhasset's master architect.
</item 7>

<item 8>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 8>

<item 9>
New Roc City
New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in
Westchester County, New York.

 It is located at 33 LeCount Place, between Main Street North and Main Street South.

 The center was built on the site of the former New Rochelle Mall which closed in 1992.
</item 9>

<item 10>
Luxify
Luxify is a Hong Kong-based company, providing an online marketplace for new, vintage and pre-owned luxury goods.

 The company manages an online marketplace website in which members buy and sell a variety of luxury goods.

 It was Hong Kong's first online marketplace to buy and sell luxury goods The site has over 15 categories, including real estate, watches, jewelry, luxury cars,
yachts, aircraft, luxury handbags, art, collectibles, furniture, antiques and fine wines & spirits.

 As of March 2016, the website had over 22,000 listings from luxury dealers from all over the world, making a US$18.6 billion marketplace.

 Luxify also has offices in Singapore and London and local partners in Shanghai and Jakarta.
</item 10>
=== END ITEM id=d85ff0fdae24096f ===

=== ITEM id=440c0cbf36f7be26 ===
Gold sentences and facts:
Sentence 0:  He is owner and chef of "Dear.
  Fact 0.0: He is the owner of Dear.
  Fact 0.1: He is the chef of Dear.
Sentence 1:  It airs on JTBC on Mondays at 21:30 (KST) beginning November 17, 2014.
  Fact 1.0: The show airs on JTBC.
  Fact 1.1: The show airs on Mondays at 21:30 (KST).
  Fact 1.2: The show begins airing on November 17, 2014.

Retrieved items:
<item 1>
Ms Panda and Mr Hedgehog
Panda and Hedgehog () is a 2012 South Korean romantic comedy television series starring Lee Donghae of Super Junior and Yoon Seung-ah.

 Produced by Song Hae-sung's TV production venture Lion Fish, it aired on Channel A from August 18 to October 7, 2012 on Saturdays and Sundays at 21:55 for 16
episodes.
</item 1>

<item 2>
Hidden Identity (TV series)
Hidden Identity () is a 2015 South Korean television series starring Kim Bum, Park Sung-woong, Yoon So-yi and Lee Won-jong.

 It aired on tvN from June 16 to August 4, 2015 on Mondays and Tuesdays at 23:00 for 16 episodes.
</item 2>

<item 3>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 3>

<item 4>
Door to Door (Modern Family)
"Door to Door" is the fourth episode of the third season of the American sitcom "Modern Family", and the series' 52nd episode overall.

 "Door to Door" first aired on October 5, 2011, on ABC.

 The episode was written by Bill Wrubel and directed by Chris Koch.
</item 4>

<item 5>
Cook Representative
Cook Representative (), also known as National Chef Team, is a 2016 South Korean cooking-variety program starring Kim Sung-joo, Ahn Jung-hwan, Kang Ho-dong,
Choi Hyun-seok, Sam Kim, Lee Won-il and, , is the spin-off of "Please Take Care of My Refrigerator".

 It aired on JTBC during Wednesdays at 22:50 (KST) beginning February 17, 2016.

 The series aired its last episode August 10, 2016 after concluding the finals match of the World Championship.
</item 5>

<item 6>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 6>

<item 7>
Black (TV series)
Black () is an upcoming fantasy-thriller South Korean television series starring Song Seung-heon, Go Ara, Lee El and Kim Dong-jun.

 It is set to air on OCN starting October 14, 2017 on Saturdays and Sundays at 22:20 (KST) time slot, replacing "Save Me".
</item 7>

<item 8>
Please Take Care of My Refrigerator
Please Take Care of My Refrigerator () is a 2014 South Korean cooking-variety program starring by many chefs and celebrity guests.

 It airs on JTBC on Mondays at 21:30 (KST) beginning November 17, 2014.
</item 8>

<item 9>
Lee Won-il
Lee Won-il (; born December 12, 1979) is a South Korean chef and television personality.

 He is currently a cast member in the variety show "Please Take Care of My Refrigerator".

 He is owner and chef of "Dear.

 bread".
</item 9>

<item 10>
Baek Jong-won's Food Truck
Baek Jong-won's Food Truck () is a South Korean cooking-variety program which has been broadcast since August 28, 2015.

 The program is currently hosted by Baek Jong-won and Kim Sung-joo, it airs on SBS every Friday at 23:20 (KST).
</item 10>
=== END ITEM id=440c0cbf36f7be26 ===

=== ITEM id=27b98ccf07235fd0 ===
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
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 1>

<item 2>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 2>

<item 3>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 3>

<item 4>
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
</item 4>

<item 5>
Edmund Mortimer (1302–1331)
Sir Edmund Mortimer (1302/1303 – 16 December 1331) was the eldest son of Roger Mortimer, 1st Earl of March and Joan de Geneville, 2nd Baroness Geneville.

 By his wife Elizabeth de Badlesmere he was the father of Roger Mortimer, 2nd Earl of March.

 Though Edmund survived his father by one year, he did not inherit his father's lands and titles as they were forfeited to the Crown and his son only reacquired
them gradually.
</item 5>

<item 6>
Edmund Mortimer, 2nd Baron Mortimer
Edmund Mortimer, 2nd Lord Mortimer (1251 – 17 July 1304) was the second son and eventual heir of Roger Mortimer, 1st Baron Mortimer.

 His mother was Maud de Braose.

 As a younger son, Edmund had been intended for clerical or monastic life, and had been sent to study at Oxford University.
</item 6>

<item 7>
AES+F
AES+F is an artist collective of four Russian artists: Tatiana Arzamasova (1955), Lev Evzovich (1958), Evgeny Svyatsky (1957), and Vladimir Fridkes (1956).

 It was first formed as AES Group in 1987 by Arzamasova, Evzovich, and Svyatsky, becoming AES+F when Fridkes joined in 1995.

 The collective works in photography, video, installation, and animation, as well as more traditional media, such as painting, drawing, and sculpture.

 AES+F's early work included performance, installation, painting, and illustration.

 Well known for their monumental video-art installations that Gareth Harris describes as "monumental painting set in motion", AES+F create grand visual
narratives that explore contemporary global values, vices and conflicts.
</item 7>

<item 8>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 8>

<item 9>
Scott Free Productions
Scott Free Productions is a British film and television production company founded by filmmakers and brothers Ridley Scott and Tony Scott.

 They formed the feature film development company Percy Main Productions in 1980, naming the company after the English village Percy Main, where their father
grew up.

 The company was renamed to Scott Free Productions in 1995.

 Scott Free has produced films ranging from the 2000 Hollywood blockbuster "Gladiator" (2000) to "smaller pictures" like "Cracks" (2009).

 Between productions of "White Squall" (1996) and "G.I. Jane" (1997), Ridley Scott reorganized the company.
</item 9>

<item 10>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 10>
=== END ITEM id=27b98ccf07235fd0 ===

=== ITEM id=3161936277705c34 ===
Gold sentences and facts:
Sentence 0: Chalk and Cheese was a short lived sitcom television show, featuring Michael Crawford and Robin Hawdon.
  Fact 0.0: Chalk and Cheese was a short-lived sitcom television show
  Fact 0.1: Chalk and Cheese featured Michael Crawford
  Fact 0.2: Chalk and Cheese featured Robin Hawdon
Sentence 1: Michael Patrick Smith, {'1': ", '2': ", '3': ", '4': "} (born 19 January 1942), known as Michael Crawford, is an English actor, comedian and singer.
  Fact 1.0: Michael Patrick Smith is known as Michael Crawford
  Fact 1.1: Michael Patrick Smith was born 19 January 1942
  Fact 1.2: Michael Crawford is English
  Fact 1.3: Michael Crawford is an actor
  Fact 1.4: Michael Crawford is a comedian
  Fact 1.5: Michael Crawford is a singer

Retrieved items:
<item 1>
Chalk and Cheese
Chalk and Cheese was a short lived sitcom television show, featuring Michael Crawford and Robin Hawdon.
</item 1>

<item 2>
Michael Crawford
Michael Patrick Smith, {'1': ", '2': ", '3': ", '4': "} (born 19 January 1942), known as Michael Crawford, is an English actor, comedian and singer.

 He has received great critical acclaim and won numerous awards during his career, which covers radio, television, film, and stagework on both London's West End
and on Broadway in New York City.
</item 2>

<item 3>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 3>

<item 4>
Roger Kitter
Roger Kitter (20 October 1949 – 3 January 2015) was an English actor best known for playing Captain Alberto Bertorelli in series 7 of the British sitcom TV
series "'Allo 'Allo!

".

 He had previously appeared weekly with Lulu throughout the 10-week run of her 1973 BBC1 series "It's Lulu".

 With Kaplan Kaye he also recorded a song "Chalk Dust – the Umpire Strikes Back" using the moniker 'The Brat'.

 Released on the Hansa label, it entered the UK Singles Chart on 10 July 1982; it reached a peak of number 19, and remained in the chart for 8 weeks.

 The song was a Top 10 hit in the Netherlands, Belgium and South Africa, and lampooned John McEnroe complaining about line calls in tennis ("The ball's in,
everyone can see that the ball's in!")

.

 He was also an impressionist, who provided the voice of Tommy Cooper in the Lego 'Kipper' advertisement.
</item 4>

<item 5>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 5>

<item 6>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 6>

<item 7>
A Cream Cracker under the Settee
"A Cream Cracker Under The Settee" is a dramatic monologue written by Alan Bennett in 1987 for television, as part of his "Talking Heads" series for the BBC.

 The series became very popular, moving onto BBC Radio, international theatre, becoming one of the best-selling audio book releases of all time and included as
part of both the A-level and GCSE English syllabus.

 It was the sixth and final episode of the first series of "Talking Heads".
</item 7>

<item 8>
Jonathan Routh
Jonathan Routh, born John Reginald Surdeval Routh, (1927–2008) co-starred in the British version of the television show "Candid Camera" (1960–67) and co-starred
with Germaine Greer and Kenny Everett in a later attempt at a revival, "Nice Time" (1968).

 He published a number of humorous books, and also painted for many years.
</item 8>

<item 9>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 9>

<item 10>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 10>
=== END ITEM id=3161936277705c34 ===

=== ITEM id=82e7b195b2022881 ===
Gold sentences and facts:
Sentence 0: Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.
  Fact 0.0: Paul Clyne was the District Attorney of Albany County, New York.
  Fact 0.1: Paul Clyne held this office from January 2001.
  Fact 0.2: Paul Clyne held this office through December 2004.
Sentence 1: P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.
  Fact 1.0: P. David Soares was born on October 26, 1969.
  Fact 1.1: P. David Soares was born in Brava, Cape Verde.
  Fact 1.2: P. David Soares is the District Attorney of Albany County, New York.

Retrieved items:
<item 1>
Paul Clyne
Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.

 A graduate of Albany Law School, he spent about 14 years as an assistant district attorney, before he was tapped by local politicians to replace the retiring
District Attorney, Sol Greenberg.

 He was defeated for re-election by David Soares, first in the Democratic Party primary election in September 2004, and then in the general election in November
2004, in which he ran on an independent line.

 After a stint teaching at the New York Prosecutors Institute, he went into private practice as a criminal defense attorney in 2007, with an office in Albany,
New York.
</item 1>

<item 2>
Richard Pleasance
Richard Arnold Pleasance is an Australian rock musician and producer.

 He was a founding member of Boom Crash Opera on guitar, bass guitar, vocals and as a songwriter in 1985; they released three albums before Pleasance left in
1992.

 Their hit Australian Recording Industry Association (ARIA) singles, "Great Wall" (No. 5, 1986) and "Onion Skin" (No. 11, 1989) were co-written by Pleasance,
who also co-produced their second album, "These Here Are Crazy Times" (No. 10, 1989).

 His debut solo release, "Galleon" received four nominations at the ARIA Music Awards for 1992.

 Pleasance composed the theme music for Australian television series, "SeaChange" (1998–2001), for 2006 feature film "Kenny" and more recently he composed the
theme music for the prison drama series "Wentworth".

 Pleasance is married to Michelle and, as from May 2009, he was living in Hepburn Springs, Victoria where he has a recording studio.
</item 2>

<item 3>
Câmara de Lobos
Câmara de Lobos (] ; literally, Portuguese: "chamber of the sealions" ) is a municipality, parish and town in the south-central coast of the island of Madeira.

 Technically a suburb of the much larger capital city of Funchal, it is one of the larger population centres and an extension of the Funchal economy.
</item 3>

<item 4>
Stewart Spiers
Stewart Spiers was a small but innovative firm of plane-makers in Scotland, founded first of all in Ayr in Ayrshire and continuing under the registered name of
Stewart Speirs Ltd ["sic"] in Paisley, Renfrewshire, from c. 1933 until its demise in the mid to late 1930s.

 Like the Glasgow firm of Alexander Mathieson & Sons, Spiers benefited hugely from the thriving industries on the Firth of Clyde in the latter half of the
nineteenth century.
</item 4>

<item 5>
Two Rivers Press
Two Rivers Press is an independent publishing house, based in the English town of Reading.

 Two Rivers Press was founded in 1994 by Peter Hay (1951–2003), a local artist.

 Its name reflects his enthusiasm for the town and its two rivers, the Kennet and the Thames, and its intention to explore "the place where art and history
meet".

 The name also gives a clue to the origins of the company in the, ultimately successful, opposition to Reading's proposed "Cross-Town Route", a road scheme that
would have seriously impacted the point at which the two rivers meet.
</item 5>

<item 6>
Skip Bolen
Skip Bolen is a Southern photographer of musicians, architecture, lifestyle and the culture of New Orleans.

 Born in Lafayette, Louisiana, he moved to New Orleans where he began his publishing career as a designer and art director.

 After moving to New York City, he began working at "House & Garden," renamed "HG," as Senior Designer in January 1988 with Anna Wintour and Alexander Liberman
at Condé Nast Publications.

 Spending evenings in jazz clubs, he began photographing jazz musicians in New York City and often when he regularly returned to New Orleans.

 After three years at Condé Nast Publications in New York City, he returned to New Orleans to pursue his jazz photography full-time.

 In 1998, he moved to Los Angeles where he became art director of House of Blues for seven years while photographing at night and weekends.

 He continued photographing jazz musicians and had his first major solo exhibition at the Jazz Bakery in Los Angeles on August 9, 2002.

 On July 4, 2006, he returned to New Orleans to pursue photography full-time documenting the recovery and rebuilding of New Orleans since Hurricane Katrina,
documenting the jazz scene, night-time photography and other photographic projects.
</item 6>

<item 7>
David Soares
P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.

 He is a Democrat.
</item 7>

<item 8>
Dundee Canal
The Dundee Canal was an industrial canal in Clifton and Passaic in Passaic County, New Jersey.

 It was built between 1858 and 1861 and ran parallel to the Passaic River.

 It supplied hydropower and water for manufacturing.

 There was interest by some members of the business community to modify the canal to support navigational uses, but the canal was never used for that purpose.
</item 8>

<item 9>
City Cruises
City Cruises is a public limited company that operates scheduled public sightseeing cruises on the River Thames in London, serving Westminster, London Eye,
Tower and Greenwich piers every day throughout the year.

 It operates the services mainly using purpose-built wheelchair-accessible 'RiverLiner' vessels, which have a large open top deck and enclosed lower saloon with
panoramic windows and bars, backed up by some ex-Paris single-deck conservatory-style sightseeing boats.

 The company also owns a number of other boats with which it provides a variety of private charter and waterage services on the River Thames.

 City Cruises owns the London Showboat brand, under which it operates an evening dinner, cabaret and dance cruise from Westminster Pier throughout the year, and
also owns and operates the Restaurant Ship Hispaniola, which is permanently moored alongside Victoria Embankment close to Embankment Underground station in
central London.
</item 9>

<item 10>
Paul Soniat
Paul Soniat is the Director of the New Orleans City Park Botanical Gardens.

 He was born at the old Touro Hospital in New Orleans.

 His family has been in New Orleans since 1727 and he grew up and later lived on the street bearing his family name.

 Paul is a self-taught piano player, and has released two CDs, absorbing the sights, sounds, and flavor of New Orleans.

 His first CD, titled "Born in New Orleans", was released in April 2005.

 His second CD was released after Hurricane Katrina hit the Gulf coast and the New Orleans' levees failed.

 This second CD is appropriately titled "Below the Water Line".
</item 10>
=== END ITEM id=82e7b195b2022881 ===

=== ITEM id=f5785ec2dca3da01 ===
Gold sentences and facts:
Sentence 0: Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.
  Fact 0.0: Henderson Point is a census-designated place (CDP).
  Fact 0.1: Henderson Point is in Harrison County, Mississippi, United States.
Sentence 1:  Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
  Fact 1.0: Gulfport is a city adjoining Biloxi.
  Fact 1.1: Biloxi is a county seat of Harrison County.

Retrieved items:
<item 1>
Hendersonville, North Carolina
Hendersonville is a city in Henderson County, North Carolina United States.

 This city is 22 miles south of Asheville.

 It is the county seat of Henderson County.

 Like the county, the city is named for 19th-century North Carolina Supreme Court Chief Justice Leonard Henderson.
</item 1>

<item 2>
Henderson Point, Mississippi
Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.

 It is part of the Gulfport–Biloxi Metropolitan Statistical Area.

 The population was 170 at the 2010 census.
</item 2>

<item 3>
Jasper County Community Unit School District 1
Craig Carr is the principal of this school.

 Ste. Marie Elementary School is located in the southern Jasper County village of Ste. Marie, and serves students in grades one through six.

 The principal of Ste. Marie is David Parker, the principal of Willow Hill Elementary School.

 Newton Elementary School is located in the county seat of Newton, and serves most of west Jasper County's first through sixth graders under principal Travis
Wyatt.

 The latter three elementary schools feed into Jasper County Junior High School and are taught in the facility during seventh and eighth grade while being
supervised by Newton Elementary School principal Travis Wyatt before graduating into Newton Community High School.

 Students in grades nine through twelve spend their last leg of precollegiate education at this school; their principal is Ruth Kerner.

 The district superintendent is Ron Alburtus, and the district's mascot is the eagle.
</item 3>

<item 4>
Paulding, Mississippi
Paulding is an unincorporated community in and one of the two county seats of Jasper County, Mississippi.

 It is the only unincorporated county seat in Mississippi.

 Settled in 1833, it was named by United States settlers in honor of Revolutionary War hero John Paulding.

 After its citizens refused to contribute to a new railroad, the community was bypassed in favor of Bay Springs, Mississippi, which was designated a railroad
stop to the west and the second county seat.

 It attracted more development and industry.
</item 4>

<item 5>
Henderson, Texas
Henderson is a city in Rusk County, northeast Texas, United States.

 The population was 13,712 at the 2010 census.

 It is the county seat of Rusk County.

 Henderson is named for James Pinckney Henderson, the first governor of Texas.
</item 5>

<item 6>
Raeford, North Carolina
Raeford is a city in Hoke County, North Carolina, United States.

 The population was 4,626 at the 2010 census.

 It is the county seat of Hoke County.

 The county was named after Confederate General Robert F. Hoke, a North Carolina native.
</item 6>

<item 7>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 7>

<item 8>
Nashville Terminal Subdivision
The Nashville Terminal Subdivision is a railroad line owned by CSX Transportation in the U.S. State of Tennessee.

 The Subdivision is broken up into 5 sections all in Nashville, Tennessee.

 The northern part of the Terminal is in Madison, Tennessee at milepost 000/0BA 174 on the southern end of the Ex-L&N Mainline Subdivision at Monfort.

 Disptach for the Mainline Sub is known as "LD" which is part of the Cincinnati Division.

 From here in Madison begins the double track that stays for another 22 miles south to Brentwood, Tennessee.

 At milepost 000/0BA/00H 176.6, the famous Johnny Cash "Amqui" location where the Ex-L&N Evansville, Indiana line, the Henderson Subdivision meets with the
Terminal.

 Dispatch for the Henderson Sub is known as "SA" and operates on AAR58.

 There are two crossings near each other, Williams Ave and Nesbitt Lane at Amqui.
</item 8>

<item 9>
Piney Woods Country Life School
The Piney Woods Country Life School (or The Piney Woods School) is a co-educational independent historically African-American boarding school for grades 9-12 in
Piney Woods, unincorporated Rankin County, Mississippi.

 It is 21 mi south of Jackson.

 It is one of four remaining historically African-American boarding schools in the United States.

 It is currently the largest African-American boarding school, as well as being the second oldest continually operating African-American boarding school.
</item 9>

<item 10>
Biloxi, Mississippi
Biloxi ( ) is a city in Harrison County, Mississippi, United States.

 The 2010 United States Census recorded the population as 44,054, and in 2016 the estimated population was 45,975.

 Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
</item 10>
=== END ITEM id=f5785ec2dca3da01 ===

=== ITEM id=3ef49f2de2e5e267 ===
Gold sentences and facts:
Sentence 0: The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
  Fact 0.0: The Albanian Coalition 'Perspective' is an Albanian political party.
  Fact 0.1: The Albanian Coalition 'Perspective' is in Montenegro.
Sentence 1: Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.
  Fact 1.0: Albanians in Montenegro constitute 4.91% of the county's total population.
  Fact 1.1: The Albanian name for Albanians in Montenegro is 'Shqiptarët e Malit të Zi'.

Retrieved items:
<item 1>
Darza
Darza (Albanian: "Darzë" , Montenegrin and ) is a village in the Ulcinj Municipality, southeastern Montenegro.

 It is a multi-ethnic settlement, inhabited by Montenegrins, Serbs and Albanians.

 According to the 2003 census, the total population was 119.
</item 1>

<item 2>
Albania
Albania is a unitary parliamentary constitutional republic with the capital in Tirana, the country's largest city and main economic and commercial centre,
followed by Durrës.

 The country's other major cities include Vlorë, Sarandë, Shkodër, Berat, Korçë, Gjirokastër and Fier.
</item 2>

<item 3>
Minority languages of Montenegro
European Charter for Regional or Minority Languages entered into force in Montenegro in June 2006, following the independence of Montenegro from the State Union
of Serbia and Montenegro on 3 June 2006.

 Constitution of Montenegro from 2007 states that Montenegrin language is official language of country, while Bosnian, Croatian, Serbian and Albanian are
languages in official use.

 The Constitution states that languages in official use are those of groups that form at least 1% of the population of Montenegro, as per 2003 population
census.

 The Law on National Minorities specifies that the percentage of members of national minorities in total population of the local government should be 15% in
order for their language and script to be introduced in official use.

 Media whose founder is Montenegro government are obliged to broadcast news, cultural, educational, sports and entertainment programs in minority languages.

 Minorities and their members have the right to education in their language in regular and vocational education.
</item 3>

<item 4>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 4>

<item 5>
Albanians
Albanians (Albanian: "Shqiptarët" ) are a nation and ethnic group, native to Albania, Kosovo and other countries who share a common culture, ancestry and speak
the Albanian language as a native tongue.

 Legally, the term is used to refer to the citizens of the Republic of Albania.

 Ethnic Albanians speak the Albanian language and more than half of ethnic Albanians live in Albania and Kosovo.

 A large Albanian population lives in Greece, Italy, the Republic of Macedonia, with smaller Albanian populations located in Serbia and Montenegro.
</item 5>

<item 6>
Albanians in the Republic of Macedonia
Albanians are the largest ethnic minority in the Republic of Macedonia.

 Of the 2,022,547 citizens of Macedonia, 509,083, or 25.2%, are Albanian according to the latest national census in 2002.

 The Albanian minority lives mostly in the north-western part of the country.

 The largest Albanian communities are in the municipalities of Tetovo (70.3% of the total population), Gostivar (66.7%), Debar (58.1%), Struga (56.8%), Kičevo
(54.5%), Kumanovo (25.8%) and Skopje (20.4%).
</item 6>

<item 7>
Albanian Coalition &quot;Perspective&quot;
The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
</item 7>

<item 8>
Graph query row:
organization: Albanian Coalition "Perspective"
kind: political party
relationship: LOCATED_IN
place: Montenegro
country: None
</item 8>

<item 9>
Albania
Albania ( , ; Albanian: "Shqipëri/Shqipëria" ; Gheg Albanian: "Shqipni/Shqipnia or Shqypni/Shqypnia" ), officially the Republic of Albania (Albanian: "Republika
e Shqipërisë" , ] ), is a country in Southern and Southeastern Europe.

 The country spans 28,748 km2 and had a total population of almost 3 million people as of 2016 .

 Albania is located in the southwestern part of the Balkan Peninsula, bordered by Montenegro to the northwest, Kosovo to the northeast, the Republic of
Macedonia to the east, and Greece to the south and southeast.

 The country has a coastline on the northern shore of the Mediterranean Sea, the Adriatic Sea to the west and the Ionian Sea to the southwest, forming the
Albanian Riviera.

 Albania is less than 72 km from Italy, across the Strait of Otranto which connects the Adriatic Sea to the Ionian Sea.
</item 9>

<item 10>
Albanians in Montenegro
Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.

 Albanians of Montenegro are Ghegs who mainly live in southeastern and eastern Montenegro, mainly in the following municipalities: Ulcinj (71% of population),
Plav (19%), Bar (6%), Podgorica (5%) and Rožaje (5%).
</item 10>
=== END ITEM id=3ef49f2de2e5e267 ===

=== ITEM id=5ee6d18fc3c207e9 ===
Gold sentences and facts:
Sentence 0: During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.
  Fact 0.0: Operation Cold Comfort was a failed SAS raid.
  Fact 0.1: Operation Cold Comfort took place during World War II.
  Fact 0.2: Operation Cold Comfort began with a parachute drop.
  Fact 0.3: The parachute drop of Operation Cold Comfort was north of Verona.
  Fact 0.4: The parachute drop of Operation Cold Comfort was on February 17, 1945.
Sentence 1:  The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.
  Fact 1.0: The SAS was founded in 1941 as a regiment.
  Fact 1.1: The SAS was later reconstituted as a corps.
  Fact 1.2: The SAS was reconstituted as a corps in 1950.

Retrieved items:
<item 1>
Joint Special Operations Command (Jordan)
The Special Operation Forces of the Jordanian Armed Forces serve as Jordan's premiere special forces unit.

 Founded on April 15, 1963 on the orders of the late King Hussein, its primary roles include reconnaissance, counter-terrorism, search and evacuation,
intelligence gathering combat, and the protection of key sites.

 The Special Operation Forces are also charged with carrying out precision strikes against critical enemy targets.

 The 14,000-strong unit are equipped and trained to be able to operate behind enemy lines for long periods without any logistical support, and is considered one
of the finest special forces units in the world.
</item 1>

<item 2>
Special Operations Force (Singapore)
The Special Operations Force (SOF) is the Republic's Army Special Forces composed of highly trained elite soldiers within the Singapore Armed Forces Commando
Formation and an essential component of the joint special forces unit Special Operations Task Force (SOTF).

 According to the Principles of Special Forces, the Republic's special forces thrive on their exceptional qualities and advanced skills, consisting highly
adaptive individuals who are independent and can operate independently, and the Republic's special forces soldiers cannot be mass-produced and must be managed
carefully.
</item 2>

<item 3>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 3>

<item 4>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 4>

<item 5>
Operation Cold Comfort
During World War II, Operation Cold Comfort was a failed SAS raid that began with a parachute drop north of Verona on February 17, 1945.

 It was later renamed Zombie.
</item 5>

<item 6>
Jan Breytenbach
Jan Dirk Breytenbach {'1': ", '2': ", '3': ", '4': "} (born 4 July 1933) was appointed by General Fritz Loots, the founder of the South African Special Forces
Brigade, as the first commander of 1 Reconnaissance Commando, the first unit founded within the South African Special Forces.

 He was also appointed as the first commander of the 32 Battalion, known colloquially as "Buffalo Battalion", as well as 44 Parachute Brigade.
</item 6>

<item 7>
Cold Comfort (Inside No. 9)
"Cold Comfort" is the fourth episode of the second series of the British dark comedy anthology television programme "Inside No. 9".

 The episode, which was written and directed by Steve Pemberton and Reece Shearsmith, was first broadcast on 16 April 2015 on BBC Two.

 Most of "Cold Comfort" is composed of a stream from a fixed camera on the desk of Andy, the protagonist, with smaller pictures on the side of the screen, in
the style of a CCTV feed.

 "Cold Comfort" was filmed over two and a half days in Twickenham, and was, like "A Quiet Night In" from "Inside No. 9"'s first series, highly experimental.

 It was Pemberton and Shearsmith's directorial debut.
</item 7>

<item 8>
Orlov let
Orlov let ("Eagle's Flight") was the name of an anti-terrorist operation conducted by the Montenegrin police to arrest a group of Albanians who planned
terrorist attacks and an armed conflict in Albanian-inhabited parts of Montenegro.

 The group of 17 people planned operations from 2004 to their arrest on the night of 10/11 September 2006.

 Weapons and explosive material were found in police raids.

 The group's main operative was the destroying of cultural and religious buildings (Orthodox) in Albanian-inhabited territory.

 The group was organised by an Albanian association based in Detroit, and members of the Kosovo Liberation Army, from which they received financial aid and
smuggled arms into Montenegro.

 The group was given a total sentence of 51 years of prison.

 Relations between the state and the Albanian minority strained after the event.
</item 8>

<item 9>
Special Air Service
The Special Air Service (SAS) is a special forces unit of the British Army.

 The SAS was founded in 1941 as a regiment, and later reconstituted as a corps in 1950.

 The unit undertakes a number of roles including covert reconnaissance, counter-terrorism, direct action and hostage rescue.
</item 9>

<item 10>
Operation Barras
Operation Barras was a British Army operation that took place in Sierra Leone on 10 September 2000, during the late stages of that nation's civil war.

 The operation aimed to release five British soldiers of the Royal Irish Regiment who had been held by a militia group known as the "West Side Boys".

 The soldiers were part of a patrol that was returning from a visit to Jordanian peacekeepers attached to the United Nations Mission in Sierra Leone (UNAMSIL)
at Masiaka on 25 August 2000 when they turned off the main road and down a track towards the village of Magbeni.

 There the patrol was overwhelmed by a large number of heavily armed rebels, taken prisoner, and transported to Gberi Bana on the opposite side of Rokel Creek.
</item 10>
=== END ITEM id=5ee6d18fc3c207e9 ===

=== ITEM id=a95a8455c7210c33 ===
Gold sentences and facts:
Sentence 0: That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.
  Fact 0.0: That Darn Cat is a film.
  Fact 0.1: That Darn Cat is from 1997.
  Fact 0.2: That Darn Cat is American.
  Fact 0.3: That Darn Cat is a mystery comedy.
  Fact 0.4: That Darn Cat stars Christina Ricci.
  Fact 0.5: That Darn Cat stars Doug E. Doug.
Sentence 1: Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".
  Fact 1.0: Honey, I Blew Up the Kid is a film.
  Fact 1.1: Honey, I Blew Up the Kid is from 1992.
  Fact 1.2: Honey, I Blew Up the Kid is American.
  Fact 1.3: Honey, I Blew Up the Kid is a comedy science fiction film.
  Fact 1.4: Honey, I Blew Up the Kid is a sequel to Honey, I Shrunk the Kids.
  Fact 1.5: Honey, I Shrunk the Kids is a 1989 film.

Retrieved items:
<item 1>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 1>

<item 2>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 2>

<item 3>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 3>

<item 4>
Honey, We Shrunk Ourselves
Honey, We Shrunk Ourselves is a 1997 live-action direct-to-video sequel to "Honey, I Shrunk the Kids" and "Honey, I Blew Up the Kid".

 It is the third and final installment in the "Honey, I Shrunk the Kids" trilogy.

 The directorial debut of cinematographer Dean Cundey and released through Walt Disney Home Video, it tells the story of inventor Wayne Szalinski as he
accidentally shrinks his wife, brother, sister-in-law, and himself with his electromagnetic shrink ray.
</item 4>

<item 5>
Off His Rockers
Off His Rockers is an animated short film produced by Walt Disney Feature Animation and released in 1992.

 It was released theatrically accompanying the film "Honey, I Blew Up the Kid".

 The film started as a side project of director Barry Cook, who, at the time, was working at Walt Disney Feature Animation Florida.

 Starting with a core of six people, the project ended up involving most of the Florida studio's staff of 73 (and some in California), all of whom "donated"
their time to the project beyond their official duties.
</item 5>

<item 6>
The Apple Dumpling Gang
The Apple Dumpling Gang is a 1971 novel by Jack Bickham, about a group of orphaned children during the California gold rush.

 They encounter a gambler who reluctantly helps them, as well as a pair of hapless robbers who are after the gold the children have found.
</item 6>

<item 7>
List of Tottoko Hamutaro Hai! episodes
Tottoko Hamutaro Haai!

 (とっとこハム太郎は〜い!

 , Tottoko Hamutarō Hai! )

 is the sequel to 'Tottoko Hamutaro: Hamu Hamu Paradichu!

.

 The series revolves around the 15 original Ham-Hams in short 5-minute episodes.

 The Hai!

 series is animated differently from the original series, most notably, the head-to-body ratio is off, and it is also computer-generated in 3-D.

 It has never been aired in English, but has aired in Italian under the title of "Hi!

 Hamtaro: Piccoli Criceti Grandi Avventure", in Mandarin under the same name as the original series, "Hamutailang" (Hamtaro), as well as in Thai and Korean, and
in Indonesian under "Hi!

 Hamtaro: Little Hamsters, Big Adventures (its official English name) .

 There is a video game called "Tottoko Hamutaro Hai!

: Hamu Hamu Challenge!

 Atsumare Hai!"
</item 7>

<item 8>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 8>

<item 9>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 9>

<item 10>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 10>
=== END ITEM id=a95a8455c7210c33 ===

=== ITEM id=5d7a352c7e4b5a1a ===
Gold sentences and facts:
Sentence 0: Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.
  Fact 0.0: Hou Hsiao-hsien was born on 8 April 1947.
  Fact 0.1: Hou Hsiao-hsien is a Taiwanese film director.
  Fact 0.2: Hou Hsiao-hsien is a Taiwanese screenwriter.
  Fact 0.3: Hou Hsiao-hsien is a Taiwanese producer.
  Fact 0.4: Hou Hsiao-hsien is a Taiwanese actor.
Sentence 1:  He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.
  Fact 1.0: Hou Hsiao-hsien is a leading figure in world cinema.
  Fact 1.1: Hou Hsiao-hsien is a leading figure in Taiwan's New Wave cinema movement.
  Fact 1.2: Hou Hsiao-hsien is an auteur.
Sentence 2:  He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in 2015 for "The Assassin" (2015).
  Fact 2.0: Hou Hsiao-hsien won the Golden Lion at the Venice Film Festival in 1989.
  Fact 2.1: Hou Hsiao-hsien won the Golden Lion for his film "A City of Sadness" (1989).
  Fact 2.2: Hou Hsiao-hsien won the Best Director award at the Cannes Film Festival in 2015.
  Fact 2.3: Hou Hsiao-hsien won the Best Director award at Cannes for his film "The Assassin" (2015).
Sentence 3:  Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
  Fact 3.0: "The Puppetmaster" (1993) is a highly regarded work by Hou Hsiao-hsien.
  Fact 3.1: "Flowers of Shanghai" (1998) is a highly regarded work by Hou Hsiao-hsien.
Sentence 4: Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.
  Fact 4.0: Kenny Ortega (Kenneth John Ortega) was born on April 18, 1950.
  Fact 4.1: Kenny Ortega is an American producer.
  Fact 4.2: Kenny Ortega is an American director.
  Fact 4.3: Kenny Ortega is an American choreographer.
Sentence 5:  He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
  Fact 5.0: Kenny Ortega is best known for directing "Hocus Pocus".
  Fact 5.1: Kenny Ortega is best known for directing the "High School Musical" trilogy.
  Fact 5.2: Kenny Ortega is best known for directing "Descendants".
  Fact 5.3: Kenny Ortega is best known for directing Michael Jackson's "This Is It" concerts.

Retrieved items:
<item 1>
Kenny Ortega
Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.

 He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
</item 1>

<item 2>
Flight of the Red Balloon
Flight of the Red Balloon (French: Le voyage du ballon rouge ) is a 2007 French-Taiwanese film directed by Hou Hsiao-hsien.

 It is the first part in a new series of films produced by Musée d'Orsay, and tells the story of a French family as seen through the eyes of a Chinese student.

 The film was shot in August and September 2006 on location in Paris.

 This is Hou Hsiao-hsien's first non-Asian film.

 It references the classic 1956 French short "The Red Balloon" directed by Albert Lamorisse.
</item 2>

<item 3>
Academy Honorary Award
The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards (previously called the Special Award, which was first presented in early 1929) – is
given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not
covered by existing Academy Awards, although prior winners of competitive Academy Awards are not excluded from receiving the Honorary Award.

 Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.

 Unlike the Special Achievement Award instituted in 1972 (and discontinued in 1995), those on whom the Academy confers its Honorary Award do not have to meet
"the Academy's eligibility year and deadline requirements."

 Like the Special Achievement Award, the Special Award and Honorary Award have been used to reward significant achievements of the year that did not fit in
existing categories, subsequently leading the Academy to establish several new categories, and to honor exceptional career achievements, contributions to the
motion picture industry, and service to the Academy.
</item 3>

<item 4>
Hou Hsiao-hsien
Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.

 He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.

 He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in
2015 for "The Assassin" (2015).

 Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
</item 4>

<item 5>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 5>

<item 6>
Mark Lee Ping-bing
Mark Lee Ping-bing (李屏賓; born August 8, 1954) is a Taiwanese cinematographer, photographer and author with over 70 films and 21 international awards to his
credit including 2 Glory Of The Country Awards from the Government Information Office of Taiwan and the president of Taiwan's Light Of The Cinema Award.

 Lee began his film career in 1977 and in 1985 he started his prolific collaboration with Taiwanese filmmaker Hou Hsiao-hsien.

 Known best for his use of natural lighting utilizing real film and graceful camera movement, Lee received the Grand Technical Prize at the Cannes Film Festival
in 2000 for "In the Mood for Love".

 A member of the Academy of Motion Picture Arts and Sciences, Lee was honored with nominations by the American Society of Cinematographers for its 2014 First
Annual Spotlight Award for Best Cinematography for his work on the 2012 film "Renoir" and by the French Academy of Cinema Arts for a Cesar Award for Best
Cinematography in 2014 also for the film "Renoir".
</item 6>

<item 7>
The Assassin (2015 film)
The Assassin (; or: "The Assassin Niè Yǐnniáng") is a 2015 wuxia film directed by Taiwanese director Hou Hsiao-hsien.

 A Taiwan/China/Hong Kong co-production, it was an official selection in the main competition section at the 2015 Cannes Film Festival.

 At Cannes, Hou won the award for Best Director.

 It was released in China and Hong Kong on 27 August, and a day later in Taiwan on 28 August 2015.

 It was selected as the Taiwanese entry for the Best Foreign Language Film at the 88th Academy Awards but it was not nominated.
</item 7>

<item 8>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 8>

<item 9>
Dust in the Wind (film)
Dust in the Wind is a 1986 film by Taiwanese filmmaker Hou Hsiao-hsien.

 It is based on co-screenwriter Wu Nien-jen's own experiences, and is the final part of Hou's coming-of-age trilogy, with the others being "A Summer at
Grandpa's" (1984) and "The Time to Live and the Time to Die" (1985).
</item 9>

<item 10>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 10>
=== END ITEM id=5d7a352c7e4b5a1a ===

=== ITEM id=a8714e991309624b ===
Gold sentences and facts:
Sentence 0:  The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".
  Fact 0.0: The album features a pair of duets with Waylon Jennings.
  Fact 0.1: One of the duets with Waylon Jennings was the "There Ain't No Good Chain Gang" single.
  Fact 0.2: The duet "There Ain't No Good Chain Gang" was one of Johnny Cash's first collaborations with Waylon Jennings.
  Fact 0.3: Johnny Cash and Waylon Jennings recorded songs together throughout the 1980s.
  Fact 0.4: Johnny Cash and Waylon Jennings recorded a separate album entitled "Heroes".
Sentence 1:  Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
  Fact 1.0: Johnny Cash and Waylon Jennings would also work together as The Highwaymen.
  Fact 1.1: The Highwaymen included Willie Nelson.
  Fact 1.2: The Highwaymen included Kris Kristofferson.
Sentence 2: The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.
  Fact 2.0: The Highwaymen was a country music supergroup.
  Fact 2.1: The Highwaymen was composed of four of the country music genre's biggest artists.
  Fact 2.2: The four artists were well known for their pioneering influence on the outlaw country subgenre.
  Fact 2.3: The Highwaymen members were Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.
Sentence 3:  Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty Records.
  Fact 3.0: The Highwaymen was active as a group between 1985 and 1995.
  Fact 3.1: The Highwaymen recorded three major label albums.
  Fact 3.2: Two of The Highwaymen's major label albums were on Columbia Records.
  Fact 3.3: One of The Highwaymen's major label albums was for Liberty Records.

Retrieved items:
<item 1>
The Closer You Get...
The Closer You Get... is the seventh studio album by country music band Alabama, released in 1983.

 All three singles from this album — "The Closer You Get", "Lady Down on Love" and "Dixieland Delight" — reached Number One on the Hot Country Songs charts in
1983.

 "She Put the Sad in All His Songs" was also recorded by Ronnie Dunn (who formed the duo Brooks & Dunn with Kix Brooks in 1991) and was released by him as a
single in 1983.
</item 1>

<item 2>
Confederate Railroad
Confederate Railroad is an American country rock–Southern rock band founded in 1987 in Marietta, Georgia, by Danny Shirley (lead vocals, rhythm guitar), Michael
Lamb (lead guitar), Mark Dufresne (drums), Chris McDaniel (keyboards), Warren "Gates" Nichols (steel guitar) and Wayne Secrest (bass guitar).

 After serving as a backing band for outlaw country acts David Allan Coe and Johnny Paycheck, the band signed to a recording contract with Atlantic Records,
releasing their self-titled debut album that year.

 In the 1990s, they released four more albums for Atlantic.
</item 2>

<item 3>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 3>

<item 4>
Leave Them Boys Alone
"Leave Them Boys Alone" is a song recorded by American country music artist Hank Williams Jr. with Waylon Jennings and Ernest Tubb.

 It was released in May 1983 as the second single from Williams' album "Strong Stuff".

 The song reached number 6 on the "Billboard" Hot Country Singles chart.

 It was written by Williams, Dean Dillon, Gary Stewart and Tanya Tucker.

 The song is notable for its combination of two singers associated with the outlaw movement with a country legend from the honky tonk days and golden age of the
Grand Ole Opry.

 Outlaw singers like Williams and Jennings saw themselves as taking country music back to its raw, honky tonk roots, and recording an up tempo song with Tubb
(who would never have received radio airplay in the late 1970s and early 80's) and reaching #6 was a slap in the face to the proponents of the country pop
sound.

 The lyrics of the song, much like Williams' "Family Tradition" echo the sentiment that the outlaw singers and their current escapades were predated by the hard
living honky tonkers of the 1950s such as Hank Williams, Sr.
</item 4>

<item 5>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 5>

<item 6>
I Would Like to See You Again
I Would Like to See You Again is the 57th album by American country singer Johnny Cash, released on Columbia Records in 1978.

 The title track peaked at #12 on the singles chart, while "There Ain't No Good Chain Gang" reached #2; the album itself peaked at #23.

 The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first
collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".

 Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
</item 6>

<item 7>
The Boys Are Back (The Oak Ridge Boys album)
The Boys Are Back is the thirtieth studio album of country music group The Oak Ridge Boys.

 It was released in 2009 under the Spring Hill Music Group label.

 The album marked the group's return to secular country music after releasing gospel albums since 1992.

 The track "Seven Nation Army," a cover of The White Stripes's 2003 song from their album "Elephant", was released as the album's first single.
</item 7>

<item 8>
There Ain't No Good Chain Gang
"There Ain't No Good Chain Gang" is a song written by Hal Bynum and Dave Kirby, and recorded by American country music artists Johnny Cash and Waylon Jennings.

 It was released in May 1978 as the second single from the album "I Would Like to See You Again".

 The song reached #2 on the "Billboard" Hot Country Singles & Tracks chart.
</item 8>

<item 9>
The Highwaymen (country supergroup)
The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country
subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.

 Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty
Records.

 Their Columbia works produced three chart singles, including the number one "Highwayman" in 1985.
</item 9>

<item 10>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 10>
=== END ITEM id=a8714e991309624b ===
