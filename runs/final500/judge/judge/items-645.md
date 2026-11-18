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

=== ITEM id=169c90796c7b5aa3 ===
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
Lee Won-il
Lee Won-il (; born December 12, 1979) is a South Korean chef and television personality.

 He is currently a cast member in the variety show "Please Take Care of My Refrigerator".

 He is owner and chef of "Dear.

 bread".
</item 1>

<item 2>
Cook Representative
Cook Representative (), also known as National Chef Team, is a 2016 South Korean cooking-variety program starring Kim Sung-joo, Ahn Jung-hwan, Kang Ho-dong,
Choi Hyun-seok, Sam Kim, Lee Won-il and, , is the spin-off of "Please Take Care of My Refrigerator".

 It aired on JTBC during Wednesdays at 22:50 (KST) beginning February 17, 2016.

 The series aired its last episode August 10, 2016 after concluding the finals match of the World Championship.
</item 2>

<item 3>
Please Take Care of My Refrigerator
Please Take Care of My Refrigerator () is a 2014 South Korean cooking-variety program starring by many chefs and celebrity guests.

 It airs on JTBC on Mondays at 21:30 (KST) beginning November 17, 2014.
</item 3>

<item 4>
3rd APAN Star Awards
The 3rd APAN Star Awards () is an awards ceremony for excellence in television in South Korea.

 It was held at the Hall of Jeongsimhwa International Cultural Center, Chungnam National University in Daejeon on November 15, 2014 and hosted by T-ara's Park
So-yeon and Kim Sung-joo.

 The nominees were chosen from 87 Korean dramas that aired from November 1, 2013 to September 2014.
</item 4>

<item 5>
Hidden Identity (TV series)
Hidden Identity () is a 2015 South Korean television series starring Kim Bum, Park Sung-woong, Yoon So-yi and Lee Won-jong.

 It aired on tvN from June 16 to August 4, 2015 on Mondays and Tuesdays at 23:00 for 16 episodes.
</item 5>

<item 6>
Baek Jong-won's Food Truck
Baek Jong-won's Food Truck () is a South Korean cooking-variety program which has been broadcast since August 28, 2015.

 The program is currently hosted by Baek Jong-won and Kim Sung-joo, it airs on SBS every Friday at 23:20 (KST).
</item 6>

<item 7>
Ms Panda and Mr Hedgehog
Panda and Hedgehog () is a 2012 South Korean romantic comedy television series starring Lee Donghae of Super Junior and Yoon Seung-ah.

 Produced by Song Hae-sung's TV production venture Lion Fish, it aired on Channel A from August 18 to October 7, 2012 on Saturdays and Sundays at 21:55 for 16
episodes.
</item 7>

<item 8>
Open House Party
Open House Party (most of the time, shortened to OHP) is an American radio show hosted on Saturday and Sunday nights by Kannon, who also hosts the afternoon
drive slot on 102.9 Now.

 The show promotes itself as "the Biggest Party on the Planet".

 It focuses on playing contemporary hit radio (CHR) music, also known as Top 40.

 The show differentiates itself from most Top 40 because it plays a high rotation of remixes.

 Unlike most radio stations or programs, every song played on OHP is by a listener request, rather than computer systems automatically inserting songs that may
not even be popular anyways.

 The show started in 1987, and was hosted by John Garabedian from September 1987 to January 2017.

 Garabedian created the show, and hosted both Saturday and Sunday nights for a long period of time.

 Open House Party is broadcast on more than 50 stations around the world.
</item 8>

<item 9>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 9>

<item 10>
Black (TV series)
Black () is an upcoming fantasy-thriller South Korean television series starring Song Seung-heon, Go Ara, Lee El and Kim Dong-jun.

 It is set to air on OCN starting October 14, 2017 on Saturdays and Sundays at 22:20 (KST) time slot, replacing "Save Me".
</item 10>
=== END ITEM id=169c90796c7b5aa3 ===

=== ITEM id=4761986ae82b8924 ===
Gold sentences and facts:
Sentence 0: Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.
  Fact 0.0: Pinhead Gunpowder is an American punk rock band.
  Fact 0.1: Pinhead Gunpowder formed in East Bay, California, in 1990.
Sentence 1: Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.
  Fact 1.0: Billie Joe Armstrong was born on February 17, 1972.
  Fact 1.1: Billie Joe Armstrong is an American musician, singer, songwriter and actor.
  Fact 1.2: Billie Joe Armstrong is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day.
  Fact 1.3: Billie Joe Armstrong co-founded Green Day with Mike Dirnt.

Retrieved items:
<item 1>
Bill Schneider (musician)
William 'Bill' Schneider (born June 17) is an American musician.

 He has played bass in the punk rock band Pinhead Gunpowder since 1990, as well as providing vocals and writing the lyrics to the song "Backyard Flames".

 He has also been in many other bands such as The Influents, Monsula and Uranium 9v, Sawhorse and The Skinflutes.

 Bill was worked as bass tech during the Nimrod era for Mike Dirnt, Green Day's and as guitar tech on their albums "Nimrod" and "Warning", American Idiot, 21st
Century Breakdown, Uno Dos Tres, Revolution Radio and was their day to day manager from 2003-2012 for their "American Idiot" and 21st Century Breakdown tours
through Uno Dos Tres.

 He also owned "Broken Guitars", a guitar shop in Oakland, California with Billie Joe Armstrong.

 Bill Still works on the sidelines with the band as well as in the studio.
</item 1>

<item 2>
Billie Joe Armstrong
Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter,
and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.

 He is also a guitarist and vocalist for the punk rock band Pinhead Gunpowder and provides lead vocals for Green Day's side projects Foxboro Hot Tubs and The
Network.
</item 2>

<item 3>
Pinhead Gunpowder (2008 EP)
Pinhead Gunpowder (also known by fans as the West Side Highway EP) is the second eponymously titled extended play by the American punk rock band Pinhead
Gunpowder.

 It was released on August 19, 2008, through Recess Records, the group's first release on Recess.

 "Pinhead Gunpowder" marked the band's first new release in over 5 years.

 The first pressings of the extended play were on green and blue vinyl.

 All of the tracks were rereleased on the 2009 compilation "Kick Over the Traces".
</item 3>

<item 4>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 4>

<item 5>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 5>

<item 6>
Carry the Banner
Carry the Banner is the third EP by the Berkeley, California-based punk rock band Pinhead Gunpowder.

 Originally released on 10" vinyl in December 1994 through Too Many Records, the EP was reissued on CD by Lookout Records shortly after as the initial vinyl
pressing sold out quickly.

 It was the group's first release to feature Jason White on guitar/vocals, replacing Sarah Kirsch, who left the band in 1994 due to differences with Billie Joe
Armstrong after his main band Green Day signed to major label Reprise Records.
</item 6>

<item 7>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 7>

<item 8>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 8>

<item 9>
Compulsive Disclosure
Compulsive Disclosure is the second compilation album by the American punk rock band Pinhead Gunpowder.

 It was released on October 21, 2003, through Lookout!

 Records.

 The album features songs from the group's eponymously titled 2000 EP, the "Dillinger Four / Pinhead Gunpowder" split EP, the "8 Chords, 328 Words" EP and also
features re-recorded versions of the songs "2nd Street" and "At Your Funeral" (originally from "Dillinger Four / Pinhead Gunpowder").

 "Compulsive Disclosure" was re-released on CD and vinyl through Recess Records on February 12, 2010, with two unreleased tracks, "Salting Agents" and "El Lasso
Grappo".
</item 9>

<item 10>
Green Day related projects
Since 1991, some members of punk band Green Day have branched out past their "main band" and have started other projects with other musicians and have released
full-length albums and several EPs.

 Notable related projects of Green Day include Billie Joe Armstrong's Pinhead Gunpowder (which also featured Green Day's other guitarist Jason White), The
Frustrators in which Mike Dirnt plays bass, and The Network which many speculate has all three members of Green Day, although under stagenames.
</item 10>
=== END ITEM id=4761986ae82b8924 ===

=== ITEM id=7366be17b5c6588b ===
Gold sentences and facts:
Sentence 0:  Editors are based in Silver Spring, Maryland and Seoul, Korea.
  Fact 0.0: Editors are based in Silver Spring, Maryland.
  Fact 0.1: Editors are based in Seoul, Korea.
Sentence 1:  The magazine is headquartered in Baltimore, Maryland.
  Fact 1.0: The magazine is headquartered in Baltimore, Maryland.

Retrieved items:
<item 1>
Maryland General Assembly
The Maryland General Assembly is the state legislature of the U.S. state of Maryland that convenes within the State House in Annapolis.

 It is a bicameral body: the upper chamber, the Maryland State Senate, has 47 representatives and the lower chamber, the Maryland House of Delegates, has 141
representatives.

 Members of both houses serve four-year terms.

 Each house elects its own officers, judges the qualifications and election of its own members, establishes rules for the conduct of its business, and may
punish or expel its own members.
</item 1>

<item 2>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 2>

<item 3>
The Human Life Review
The Human Life Review is a quarterly journal published by the Human Life Foundation since 1975.

 It is devoted to explorations of life issues, primarily abortion, as well as neonaticide, medical genetics, prenatal testing, human cloning, fetal tissue
experimentation, euthanasia and assisted suicide, and also publishes articles dealing with more general questions of family and society.

 It was founded by James Patrick McFadden, formerly associate publisher of "National Review", who had also founded the Human Life Foundation, and is now edited
by his daughter, Maria McFadden.

 It was launched from the offices of "National Review", with the support of William F. Buckley.
</item 3>

<item 4>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 4>

<item 5>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 5>

<item 6>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 6>

<item 7>
Adventist HealthCare
Adventist HealthCare is a not-for-profit health services organization based in Gaithersburg, Maryland that employs more than 6,200 people and provides
healthcare for more than 400,000 individuals in the community each year.

 The primary service area for Adventist HealthCare is the Washington, D.C. metropolitan area.

 Despite similar names, it is not a part of the California-based Adventist Health, or the Florida-based Adventist Health System.
</item 7>

<item 8>
Adventist World
Adventist World is a monthly international magazine of the Seventh-day Adventist Church published by the Review and Herald Publishing Association.

 Editors are based in Silver Spring, Maryland and Seoul, Korea.
</item 8>

<item 9>
Seventh-day Adventist education
The Seventh-day Adventist educational system is part of the Seventh-day Adventist Church and overseen by the General Conference of Seventh-day Adventist located
in Silver Spring, Maryland.The educational system is the second-largest Christian school-system in the world, after the Roman Catholic system.
</item 9>

<item 10>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 10>
=== END ITEM id=7366be17b5c6588b ===

=== ITEM id=95cfabfede6f5053 ===
Gold sentences and facts:
Sentence 0:  Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
  Fact 0.0: Dan Hurley is the son of Bob Hurley.
  Fact 0.1: Bob Hurley is a Hall of Fame high school coach.
  Fact 0.2: Dan Hurley is the younger brother of Bobby Hurley.
  Fact 0.3: Bobby Hurley is a former Duke guard.
  Fact 0.4: Bobby Hurley is a former Sacramento Kings guard.
Sentence 1: Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.
  Fact 1.0: Robert Matthew Hurley was born on June 28, 1971.
  Fact 1.1: Robert Matthew Hurley is an American basketball coach.
  Fact 1.2: Robert Matthew Hurley is a former college player.
  Fact 1.3: Robert Matthew Hurley is a former professional player.
Sentence 2:  Hurley is currently the head coach of the Arizona State men's team.
  Fact 2.0: Hurley is currently the head coach of the Arizona State men's team.

Retrieved items:
<item 1>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 1>

<item 2>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 2>

<item 3>
2015–16 Iowa State Cyclones men's basketball team
The 2015–16 Iowa State Cyclones men's basketball team represents Iowa State University during the 2015–16 NCAA Division I men's basketball season.

 The Cyclones were coached by Steve Prohm, who was in his 1st season.

 They played their home games at Hilton Coliseum in Ames, Iowa and competed in the Big 12 Conference.
</item 3>

<item 4>
Bob Hurley
Robert Emmet Hurley, Sr. (born July 31, 1947) is a basketball coach at St. Anthony High School in Jersey City, New Jersey.

 Hurley has amassed 28 state championships and more than 1000 wins in 39 years as a coach.

 On February 2, 2011, Hurley became the tenth coach in high school history to win 1000 games.

 He is featured in the documentary "The Street Stops Here".

 Five of his teams have gone undefeated.

 On April 5, 2010, he was announced as the only coach to be inducted into the Naismith Memorial Basketball Hall of Fame that year and only the third high school
coach in history to be so honored; he was formally inducted on August 13 of that year.

 Hurley is the father of Bobby Hurley, a former All-American point guard at Duke and the head basketball coach at Arizona State, and Dan Hurley, who was hired
in February 2012 to coach the University of Rhode Island after two years of coaching at Wagner College and nine years coaching at Newark's Saint Benedict's
Preparatory School, also one of the top high school programs in the nation.
</item 4>

<item 5>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 5>

<item 6>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 6>

<item 7>
Rhode Island Rams men's basketball
The Rhode Island Rams men's basketball team is a college basketball program that competes in NCAA Division I and the Atlantic 10 Conference representing the
University of Rhode Island.

 The team is under the direction of head coach Dan Hurley, who was hired on March 20, 2012.

 The Rams play their home games at the Ryan Center (capacity 7,657).
</item 7>

<item 8>
Nate Oats
Nathanael "Nate" Oats (born October 13, 1974) is an American basketball coach.

 He is currently the head basketball coach at the State University of New York at Buffalo, following two seasons as an assistant coach under Bobby Hurley.

 Oats was named head coach on April 11, 2015 after Hurley was hired by Arizona State as head coach.

 In only his first season as head coach, he took the Buffalo Bulls to their second straight NCAA Tournament bid after winning the Mid-American Conference
Championship.
</item 8>

<item 9>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 9>

<item 10>
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 10>
=== END ITEM id=95cfabfede6f5053 ===

=== ITEM id=f746ef10fce4f29e ===
Gold sentences and facts:
Sentence 0: "What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).
  Fact 0.0: What the Water Gave Me is a song
  Fact 0.1: What the Water Gave Me is by English indie rock band Florence and the Machine
  Fact 0.2: What the Water Gave Me is from the band's second studio album Ceremonials
  Fact 0.3: The album Ceremonials is dated 2011
Sentence 1:  The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.
  Fact 1.0: Lead singer Florence Welch wrote the song What the Water Gave Me
  Fact 1.1: Francis Eg White wrote the song What the Water Gave Me
  Fact 1.2: Paul Epworth produced the song What the Water Gave Me
Sentence 2: What the Water Gave Me ("Lo que el agua me dio" in Spanish) is an oil painting by Frida Kahlo that was completed in 1938.
  Fact 2.0: What the Water Gave Me is an oil painting by Frida Kahlo
  Fact 2.1: The painting is titled Lo que el agua me dio in Spanish
  Fact 2.2: Frida Kahlo completed the painting in 1938

Retrieved items:
<item 1>
Lover to Lover
"Lover to Lover" is a song by English indie rock band Florence and the Machine.

 The song was released on 30 November 2012, as a digital download on iTunes as the fifth and final single from their second studio album "Ceremonials" (2011).

 The song was written by Florence Welch and Francis "Eg" White and produced by Paul Epworth.

 The Ceremonials Tour version of "Lover to Lover" is an edited version of the original with further production by Chris Hayden.

 The song became the band's first single to miss the top 200 in the UK Singles Chart.
</item 1>

<item 2>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 2>

<item 3>
Spectrum (Say My Name)
"Spectrum" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 It was written by lead singer Florence Welch and Paul Epworth with production being handled by Epworth.

 The band premiered the song during a concert at Brooklyn's Creators Project on 15 October 2011, prior to the release of the album.

 The album version of "Spectrum" is a downtempo orchestral pop, R&B, krautrock and disco song.

 The accompanying music video for the song premiered on 30 May 2012.

 It was directed by David LaChapelle and John Byrne.
</item 3>

<item 4>
Shake It Out
"Shake It Out" is a song by English indie rock band Florence and the Machine, released as the first official single from their second studio album,
"Ceremonials" (2011).

 It was written by Florence Welch and Paul Epworth, while production was handled by Epworth.

 The song was digitally released in Australia on 14 September 2011, and it was available in the United States on 19 October.

 It had its radio debut on XFM on 14 September 2011 in the United Kingdom.

 Welch revealed that the song was written within an hour and according to her it talked about shaking the regrets and the things that were haunting her.
</item 4>

<item 5>
Wish That You Were Here
"Wish That You Were Here" is a song by the English indie rock band Florence and the Machine, written by Florence Welch, Andrew Wyatt, and Emilie Haynie, and was
released on Island Records on 26 August 2016.

 The song was made available via digital download, and is featured on the soundtrack of the film "Miss Peregrine's Home for Peculiar Children" which was
directed by Tim Burton.

 Welch — a long-time fan of Burton's work — had expressed interest in collaborating with him long before the recording of "Wish That You Were Here" as they
shared similar artistic themes.

 Upon release, the composition was positively received by music critics and reached number 128 on the UK Singles Chart.
</item 5>

<item 6>
What the Water Gave Me (painting)
What the Water Gave Me ("Lo que el agua me dio" in Spanish) is an oil painting by Frida Kahlo that was completed in 1938.

 It is sometimes referred to as "What I Saw in the Water".
</item 6>

<item 7>
No Light, No Light
No Light, No Light is a song by English indie rock band Florence and the Machine from their second studio album, "Ceremonials" (2011).

 The song was written by band members Florence Welch and Isabella Summers while the production was handled by Paul Epworth.

 Island Records released the song as the second single from the album on 16 January 2012.

 The song was the first one written for the album in the band's tour bus in Amsterdam.

 Lyrically, in the song, Welch is expressing frustration about the state of her fragile relationship and she further tries to keep it together.

 "No Light, No Light" received positive reviews by music critics who generally praised Welch's vocals and the drum-led instrumentation.

 It was also placed on several critics' year-end lists of best singles.

 The song peaked at number fifty on the UK Singles Chart, the Irish Singles Chart and number thirty nine on the US "Billboard" Alternative Songs chart.
</item 7>

<item 8>
What the Water Gave Me (song)
"What the Water Gave Me" is a song by English indie rock band Florence and the Machine from their second studio album "Ceremonials" (2011).

 The song was written by lead singer Florence Welch and Francis "Eg" White, and produced by Paul Epworth.

 It was released on 23 August 2011 as the first promotional single from "Ceremonials".

 After the band performed the song in Berkeley, California on 12 June 2011, the studio version premiered on Florence and the Machine's official website.

 Welch decided to name the song "What the Water Gave Me" after seeing a Frida Kahlo painting of the same name.

 During an interview, she confirmed that the song was inspired by English writer Virginia Woolf.
</item 8>

<item 9>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 9>

<item 10>
Never Let Me Go (Florence and the Machine song)
"Never Let Me Go" is a song by English indie rock band Florence and the Machine from their second studio album, "Ceremonials" (2011).

 The song was written by Florence Welch and Paul Epworth while production was handled by Epworth.

 Island Records released the song as the third single from the album on 30 March 2012.

 The band first premiered the song during one of their concerts before the release of the album on 15 October 2011.

 A Clams Casino remix of the song was placed on the 12" white vinyl single which was available for pre-order on 22 March 2012.
</item 10>
=== END ITEM id=f746ef10fce4f29e ===

=== ITEM id=c6b82dabaf24f30b ===
Gold sentences and facts:
Sentence 0: Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.
  Fact 0.0: Hayley Nichole Williams was born on December 27, 1988.
  Fact 0.1: Hayley Nichole Williams is an American singer.
  Fact 0.2: Hayley Nichole Williams is an American songwriter.
  Fact 0.3: Hayley Nichole Williams is an American musician.
Sentence 1:  The band consists of Hayley Williams, Zac Farro and Taylor York.
  Fact 1.0: The band consists of Hayley Williams.
  Fact 1.1: The band consists of Zac Farro.
  Fact 1.2: The band consists of Taylor York.
Sentence 2: Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American heavy metal bands Lord Tracy and Pantera.
  Fact 2.0: Terrence Lee Glaze was born on November 29, 1964, in Columbus.
  Fact 2.1: Terrence Lee Glaze is known as Terry Lee.
  Fact 2.2: Terry Lee is a Texas-raised singer.
  Fact 2.3: Terry Lee is a Texas-raised guitarist.
  Fact 2.4: Terry Lee is most known for his work in the American heavy metal band Lord Tracy.
  Fact 2.5: Terry Lee is most known for his work in the American heavy metal band Pantera.

Retrieved items:
<item 1>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 1>

<item 2>
Stay the Night (Zedd song)
"Stay the Night" is a song by Russian-German electronic dance music producer Zedd, from the deluxe edition (2013) of his debut studio album, "Clarity " (2012).

 It features vocals from Hayley Williams, the lead singer of American rock group Paramore.

 The song was written by Zedd, Williams, Benjamin Eli Hanna, and Carah Faye.

 "Stay the Night" was released to digital retailers on September 10, 2013, by Interscope Records as the lead single off the deluxe edition of "Clarity", and the
fourth overall single from the album.
</item 2>

<item 3>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 3>

<item 4>
Hayley Williams (ice hockey)
Hayley Williams (born June 3, 1990) is an American professional ice hockey player for the Brampton Thunder of the Canadian Women's Hockey League (CWHL).

 Williams has previously played for the Buffalo Beauts in the National Women's Hockey League (NWHL).
</item 4>

<item 5>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 5>

<item 6>
Paramore discography
The American rock band Paramore has released five studio albums, three extended plays, two live albums, eighteen singles, one video album, and nineteen music
videos.

 The band was formed in Franklin, Tennessee, in 2004 by lead vocalist Hayley Williams with guitarists Josh Farro and Taylor York, bassist Jeremy Davis, and
drummer Zac Farro.

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling".

 Three singles were released from the album, but none of them charted.

 The album did not chart in the "Billboard" 200 either, although it peaked at number thirty in the "Billboard" Top Heatseekers.

 "All We Know Is Falling" received Gold certification in the United Kingdom and in July 2014 the RIAA certified the album Gold in the United States.
</item 6>

<item 7>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 7>

<item 8>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 8>

<item 9>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 9>

<item 10>
List of songs recorded by Paramore
American rock band Paramore have recorded songs for five studio albums, a box set, an extended play and two soundtrack albums.

 In 2002, at age 13, vocalist Hayley Williams moved to Franklin, Tennessee, where she met brothers Josh Farro and Zac Farro.

 The band was officially formed by Josh Farro (lead guitar and backing vocals), Zac Farro (drums), Jeremy Davis (bass guitar) and Williams (lead vocals) in
2004, with the later addition of Williams' neighbor Jason Bynum (rhythm guitar).

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling" that year.

 Three singles were released to promote the album.
</item 10>
=== END ITEM id=c6b82dabaf24f30b ===

=== ITEM id=c9781372598199d9 ===
Gold sentences and facts:
Sentence 0:  The name comes from the City of Starachowice, where the factory is located.
  Fact 0.0: The name comes from the City of Starachowice.
  Fact 0.1: The factory is located in the City of Starachowice.
Sentence 1:  It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).
  Fact 1.0: The item was sold in various countries for many years.
  Fact 1.1: The item was sold in countries beyond the Eastern Bloc.
  Fact 1.2: The item was used by the Yemen Army.
Sentence 2: The Eastern Bloc was the group of socialist states of Central and Eastern Europe, generally the Soviet Union and the countries of the Warsaw Pact.
  Fact 2.0: The Eastern Bloc was the group of socialist states of Central and Eastern Europe.
  Fact 2.1: The Eastern Bloc generally included the Soviet Union.
  Fact 2.2: The Eastern Bloc generally included the countries of the Warsaw Pact.

Retrieved items:
<item 1>
J. A. Baczewski
J. A. Baczewski was a Polish company most well known for its fine spirits such as vodka and gin.

 The factory, dating back to the late 18th century, was based in Lwów (Lviv) and until 1939 was one of two most popular Polish export goods.
</item 1>

<item 2>
Pioneer Pekao Investments
Pioneer Pekao is a Polish company managing several investment funds in Poland.

 It is the biggest and first investment funds manager on the Polish market.
</item 2>

<item 3>
Shaanxi Automobile Group
Shaanxi Automobile Group Co., Ltd. is a Chinese bus and truck manufacturer headquartered in Xi'an, Shaanxi.

 It employs around 23,000 employees.

 It manufactures heavy duty truck, buses (chassis), medium-size heavy truck and heavy duty axles for trucks.

 It utilizes Magna Steyr and MAN SE technologies.

 Its bus chassis are sold under the Eurostar Bus brand.
</item 3>

<item 4>
Leader Trucks
Leader found their niche in the heavy transport market and built as a rigid 4x4 and 6x6 configuration for tray, tipper and agitator applications, they offered
reliable machines in the industries of earth-moving and off-road construction.

 Leader was notable for being the first manufacturer to fit Caterpillar engines to trucks, to fit automatic transmissions to diesel trucks in assembly, and the
first truck manufacturer in Australia to offer disc brakes.

 Leader boasted its components to be 80 percent Australian built with only the engine, transmission and steering box imported.

 By 1980, Leader had sold its 1000th truck in Australia and had also sold trucks to New Zealand, South Africa, Hong Kong, China and Indonesia.
</item 4>

<item 5>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 5>

<item 6>
FSC Star
Fabryka Samochodów Ciężarowych "Star" was a Polish truck manufacturer.

 The name comes from the City of Starachowice, where the factory is located.

 Their first vehicle was the "Star 20" in 1948.

 The most popular product was the "Star 266".

 The 266 model offered very good quality and powerful engines for a low price.

 It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).

 For many years FSC Star was a state-owned company.

 Star is now owned by MAN AG who eliminated the brand in January 2009.
</item 6>

<item 7>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 7>

<item 8>
Volvo FH
The Volvo FH is a heavy truck range produced by Swedish truck manufacturer Volvo Trucks.

 Introduced in late 1993 as FH12 and FH16, production still continues with the now the second generation of FH range model lineup.

 FH stands for Forward control High entry where numbers denominate engine capacity in litres.

 The FH range is one of the most successful truck series ever having sold more than 400,000 units worldwide.
</item 8>

<item 9>
E-Z Polish Factory
The E-Z Polish Factory is a building designed by Frank Lloyd Wright and is located in Chicago, Illinois, United States.

 Built in 1905, the E-Z Polish Company made polish for shoes and stoves.

 It now serves as practice space for local bands and artists.
</item 9>

<item 10>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 10>
=== END ITEM id=c9781372598199d9 ===

=== ITEM id=0d94c8778efcc957 ===
Gold sentences and facts:
Sentence 0: Mastodon is an American heavy metal band from Atlanta, Georgia.
  Fact 0.0: Mastodon is an American heavy metal band.
  Fact 0.1: Mastodon is from Atlanta, Georgia.
Sentence 1:  Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals in studio (except Bill Kelliher).
  Fact 1.0: Mastodon was formed in 2000.
  Fact 1.1: Mastodon is composed of bassist Troy Sanders.
  Fact 1.2: Mastodon is composed of guitarist Brent Hinds.
  Fact 1.3: Mastodon is composed of guitarist Bill Kelliher.
  Fact 1.4: Mastodon is composed of drummer Brann Dailor.
  Fact 1.5: Troy Sanders, Brent Hinds, and Brann Dailor perform vocals in the studio.
  Fact 1.6: Bill Kelliher does not perform vocals in the studio.
Sentence 2:  Mastodon has released seven studio albums, as well as a number of other records.
  Fact 2.0: Mastodon has released seven studio albums.
  Fact 2.1: Mastodon has released a number of other records.
Sentence 3:  Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.
  Fact 3.0: Mastodon's second full-length release is "Leviathan".
  Fact 3.1: "Leviathan" is a concept album.
  Fact 3.2: "Leviathan" is based on the novel "Moby-Dick".
  Fact 3.3: "Moby-Dick" was written by Herman Melville.
Sentence 4: Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric Erlandson.
  Fact 4.0: Hole was an American alternative rock band.
  Fact 4.1: Hole formed in Los Angeles, California in 1989.
  Fact 4.2: Hole was formed by singer and guitarist Courtney Love.
  Fact 4.3: Hole was formed by lead guitarist Eric Erlandson.

Retrieved items:
<item 1>
Hole (band)
Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric
Erlandson.

 The band had a revolving line-up of bassists and drummers, their most prolific being drummer Patty Schemel, and bassists Kristen Pfaff (d. 1994) and Melissa
Auf der Maur.
</item 1>

<item 2>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
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
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 4>

<item 5>
Creed (band)
Creed is an American rock band formed in 1993 in Tallahassee, Florida.

 The band's best-known line-up consists of lead vocalist Scott Stapp, guitarist and vocalist Mark Tremonti, bassist Brian Marshall, and drummer Scott Phillips.

 Creed released two studio albums, "My Own Prison" in 1997 and "Human Clay" in 1999, before Marshall left the band in 2000.

 The band's third album, "Weathered", was released in 2001 with Tremonti handling bass before the band disbanded in 2004 due to increasing tension between
members.

 Tremonti, Marshall, and Phillips went on to found Alter Bridge while Stapp followed a solo career.
</item 5>

<item 6>
Mastodon (band)
Mastodon is an American heavy metal band from Atlanta, Georgia.

 Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals
in studio (except Bill Kelliher).

 They all perform vocals at live shows.

 Their musical style features progressive concepts and unique instrumentation.

 Mastodon has released seven studio albums, as well as a number of other records.

 The band's debut album, "Remission", released in 2002, garnered significant critical acclaim for its unique sound.

 Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.

 Three magazines awarded the record "Album of the Year" in 2004: "Revolver", "Kerrang!

" and "Terrorizer".
</item 6>

<item 7>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 7>

<item 8>
Blackdiamondskye
Blackdiamondskye was a 2010 North American concert tour by American bands Alice in Chains, Mastodon, and Deftones.

 Blackdiamondskye is a portmanteau of the three band's most recent albums: Alice in Chains' "Black Gives Way to Blue", Deftones' "Diamond Eyes", and Mastodon's
"Crack the Skye".

 The tour kicked off on September 16 in Chicago and culminated on October 16 in Paradise, Nevada.

 The tour lasted for 19 shows over the course of one month and featured shows at both outdoor amphitheatres and indoor venues.
</item 8>

<item 9>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 9>

<item 10>
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 10>
=== END ITEM id=0d94c8778efcc957 ===

=== ITEM id=f4157876809c3083 ===
Gold sentences and facts:
Sentence 0:  It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
  Fact 0.0: The described species is common across temperate parts of Eurasia, northern Africa, and the Americas.
  Fact 0.1: The described species is common across subtropical parts of Eurasia, northern Africa, and the Americas.
  Fact 0.2: The described species is naturalized in Australia.
  Fact 0.3: The described species is naturalized on various islands.
Sentence 1: Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South America.
  Fact 1.0: Echeveria is a large genus of flowering plants.
  Fact 1.1: Echeveria is in the Crassulaceae family.
  Fact 1.2: Echeveria is native to semi-desert areas of Central America.
  Fact 1.3: Echeveria is native to semi-desert areas of Mexico.
  Fact 1.4: Echeveria is native to semi-desert areas of northwestern South America.

Retrieved items:
<item 1>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 1>

<item 2>
Echeveria setosa
Echeveria setosa, the Mexican fire cracker, is a species of flowering plant in the Crassulaceae family, native to semi-desert areas of Mexico and common
throughout Puebla.
</item 2>

<item 3>
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 3>

<item 4>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 4>

<item 5>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 5>

<item 6>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 6>

<item 7>
Echeveria elegans
Echeveria elegans (Mexican snow ball, Mexican gem, white Mexican rose) is a species of flowering plant in the Crassulaceae family, native to semi-desert
habitats in Mexico.
</item 7>

<item 8>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 8>

<item 9>
Aloinopsis
Aloinopsis is a genus of ice plants from South Africa.
</item 9>

<item 10>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 10>
=== END ITEM id=f4157876809c3083 ===

=== ITEM id=747ae8c4b247f661 ===
Gold sentences and facts:
Sentence 0: James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more than 70 productions to his credit.
  Fact 0.0: James David Sharman was born on 12 March 1945.
  Fact 0.1: James David Sharman is the son of boxing tent entrepreneur Jimmy Sharman.
  Fact 0.2: James David Sharman is an Australian director.
  Fact 0.3: James David Sharman is an Australian writer for film and stage.
  Fact 0.4: James David Sharman has more than 70 productions to his credit.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born on August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is an American director.
  Fact 1.2: Ava Marie DuVernay is an American screenwriter.
  Fact 1.3: Ava Marie DuVernay is an American film marketer.
  Fact 1.4: Ava Marie DuVernay is an American film distributor.

Retrieved items:
<item 1>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 1>

<item 2>
Lowell Sherman
Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.

 At a time when it was highly unusual, he was both the actor and director on several films in the early 1930s, before completely transitioning to the role of
director.

 At the height of his career, after scoring huge successes with his direction of the films "She Done Him Wrong" and "Morning Glory" (which starred Mae West and
won her first Academy Award for Katharine Hepburn respectively), he succumbed to pneumonia after a brief illness.
</item 2>

<item 3>
Albert Sharpe
Albert Sharpe (15 April 1885 – 13 February 1970) was an Irish stage and film actor.

 His most famous roles were those of Darby O'Gill in Disney's "Darby O'Gill and the Little People" and as Finian McLonergan in the Original Broadway production
of the musical "Finian's Rainbow".

 (the film version, made in 1968, had Fred Astaire in the role.)

 On screen he played Fiona's father Andrew in the MGM musical Brigadoon.

 He was also a member of the Abbey Players.

 His last ten years were spent in retirement.

 He died in 1970 in Belfast at the age of 84.
</item 3>

<item 4>
♯JusticeForFlint
Initiated by "Creed" director Ryan Coogler and "Selma" director Ava DuVernay, the free public event was held at the Whiting Auditorium in Flint, Michigan.

 Hosted by comedian Hannibal Buress, it featured singers Janelle Monáe and Ledisi, as well as actor-activists Jesse Williams and Jussie Smollett, amongst
others.

 Surprising star guest was "King of Motown" Stevie Wonder, who teamed up with Monáe for "Hell You Talkin' Bout", a protest song she had written for the people
of Flint.
</item 4>

<item 5>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 5>

<item 6>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 6>

<item 7>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 7>

<item 8>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 8>

<item 9>
Jim Sharman
James David Sharman (born 12 March 1945), the son of boxing tent entrepreneur Jimmy Sharman, is an Australian director and writer for film and stage with more
than 70 productions to his credit.

 He is renowned in Australia for his work as a theatre director from the 1960s to the present, and is best known internationally as the director of the 1973
theatrical hit "The Rocky Horror Show", its film adaptation "The Rocky Horror Picture Show" (1975) and the film's follow-up "Shock Treatment" (1981).
</item 9>

<item 10>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 10>
=== END ITEM id=747ae8c4b247f661 ===

=== ITEM id=33c4a19d0131bb79 ===
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
Canada under British rule
Although confederation in 1867 led to an enlarged Dominion with increased autonomy over domestic affairs, Canada still remained a colony within the British
Empire and was thus subordinate to the British Parliament until the enactment of the Statute of Westminster in 1931.

 This statute recognized Canada as an independent peer coequal with the United Kingdom, and thus provided the Parliament of Canada with legislative sovereignty
over all federal matters except the power to change the constitutional laws of Canada which remained under the purview of the Parliament of the United Kingdom.

 Canada's final vestige of legal dependence on the United Kingdom was terminated in 1982 with the enactment of the Canada Act, subsequently providing Canada
with full legal sovereignty completely independent of the United Kingdom.
</item 1>

<item 2>
Constitution of Alberta
The Constitution of Alberta describes the fundamental rules under which the Canadian province of Alberta is governed.

 As is typical of all Canadian provinces, and Westminster systems more generally, Alberta's is an unwritten constitution.

 Alberta's constitution, like Britain's (on which it is modeled), includes any and all pieces of legislation, court decisions, proclamations, and conventions
which together inform how the province operates.

 Many statutes are important to understanding the governance of the province, but nowhere are they consolidated into a single document or even a list.

 The office of Attorney-General at one time suggested 23 acts which might be included, but cautioned that this was not a "definitive list".

 However, since Alberta is a part of federation, its powers are clearly delineated in law, via the Constitution of Canada.
</item 2>

<item 3>
British North America Acts
The British North America Acts 1867–1975 are the original names of a series of Acts at the core of the constitution of Canada.

 They were enacted by the Parliament of the United Kingdom and the Parliament of Canada.

 In Canada, some of the Acts were amended or repealed by the Constitution Act, 1982.

 The rest were renamed in Canada as the "Constitution Acts".

 In the United Kingdom, those Acts that were passed by the British Parliament remain under their original names.

 The term "British North America" (BNA) refers to the British colonies in North America.
</item 3>

<item 4>
Westminster system
The Westminster system is a parliamentary system of government modelled after that which developed in the United Kingdom.

 This term comes from the Palace of Westminster, the seat of the British parliament.
</item 4>

<item 5>
Governance of England
In some cases, such as top-up university tuition fees and foundation hospitals, the votes of Scottish MPs have been crucial in helping pass legislation for
England that the majority of English MPs have opposed.
</item 5>

<item 6>
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
</item 6>

<item 7>
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
</item 7>

<item 8>
Quebec Legislature
The Quebec Legislature (officially Parliament of Quebec, French: "Parlement du Québec" ) is the legislature of the province of Quebec, Canada, since
Confederation in 1867.

 Today, the legislature is made of two elements: the monarch of Canada, represented by the Lieutenant Governor of Quebec, and the unicameral assembly called the
National Assembly of Quebec.

 The legislature has existed since 1867 when Quebec, then called Canada East, became one of the founding colonies of Canadian Confederation.

 From 1867 until 1968 the legislature was bicameral, containing a lower chamber called the Legislative Assembly of Quebec and an upper chamber called the
Legislative Council of Quebec.
</item 8>

<item 9>
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
</item 9>

<item 10>
The New Canada
The New Canada is a Canadian political literature book written by Reform Party of Canada founder and leader Preston Manning and published by Macmillan Canada.

 The book explains the personal, religious, and political life of Preston Manning and explains the roots and beliefs of the Reform Party.

 At the time of its publishing in 1992, Reform had become a popular populist conservative party in Western Canada after the mainstream Progressive Conservative
Party of Canada was collapsing in support and in 1991 decided to expand eastward into Ontario and the Maritimes.

 One year later the PC party collapsed in the 1993 federal election, allowing the Reform Party to make political history in Canada, deplacing the PCs as the
dominant conservative party in Canada.

 Reform, later renamed the Canadian Alliance, merged with the PC Party in 2003, to form a united right-wing alternative to the governing Liberal Party of
Canada, named the Conservative Party of Canada which has dropped many of the populist themes that the Reform Party had.
</item 10>
=== END ITEM id=33c4a19d0131bb79 ===
