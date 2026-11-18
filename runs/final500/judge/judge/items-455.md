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

=== ITEM id=d8a3a5d2628d5486 ===
Gold sentences and facts:
Sentence 0: John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.
  Fact 0.0: John Alexander was born on 5 October 1955.
  Fact 0.1: John Alexander is English.
  Fact 0.2: John Alexander is a former footballer.
  Fact 0.3: John Alexander holds the post of club secretary at Manchester United.
Sentence 1:  Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in 1976.
  Fact 1.0: John Alexander was born in Liverpool.
  Fact 1.1: John Alexander began his football career with a club called Ulysses.
  Fact 1.2: John Alexander was later picked up by Millwall.
  Fact 1.3: John Alexander made his league debut for Millwall in 1976.
Sentence 2:  He is the uncle of Liverpool player Trent Alexander-Arnold.
  Fact 2.0: John Alexander is the uncle of Liverpool player Trent Alexander-Arnold.
Sentence 3: Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.
  Fact 3.0: Trent John Alexander-Arnold was born on 7 October 1998.
  Fact 3.1: Trent John Alexander-Arnold is English.
  Fact 3.2: Trent John Alexander-Arnold is a professional footballer.
  Fact 3.3: Trent John Alexander-Arnold plays as a right-back.
  Fact 3.4: Trent John Alexander-Arnold plays for Premier League club Liverpool.

Retrieved items:
<item 1>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 1>

<item 2>
Wilton Lines
Wilton Lines (fl.

 1889–1890) was an English professional footballer born in Birmingham.

 Lines played seven games for Small Heath in the inaugural 1889–90 season of the Football Alliance, covering a variety of forward positions, but was unable to
displace the established forwards.
</item 2>

<item 3>
John Alexander (footballer, born 1955)
John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.

 Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in
1976.

 After scoring twice in 15 appearances over the course of two years with Millwall, he moved to Reading; he enjoyed greater success with Reading, scoring nine
goals in 25 games.

 In 1981, he joined Northampton Town, but spent just one season with them before retiring from football at the age of 26 in 1982.

 Upon retiring from football, Alexander got a job with the BBC, but he later returned to football as club secretary at Watford.

 In 2000, he took up the same post at Tottenham Hotspur, spending 10 years there before applying for the same job at Manchester United, where he would replace
the retiring Ken Ramsden.

 He took over at Manchester United on 1 July 2010.

 He is the uncle of Liverpool player Trent Alexander-Arnold.
</item 3>

<item 4>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 4>

<item 5>
Tommy Bell (footballer, born 1906)
Thomas Bell (9 November 1906 – 1983) was an English professional footballer born in Seaham Harbour.

 He could play at inside right, centre forward and right half and was a regular goalscorer throughout his professional career.
</item 5>

<item 6>
Nick Powell
Nicholas Edward "Nick" Powell (born 23 March 1994) is an English professional footballer who plays for Wigan Athletic as a midfielder.

 He began his career at Crewe Alexandra, initially as a forward, making his debut at the age of 16.

 After impressing for Crewe Alexandra during the 2011–12 season, including scoring in their 2–0 victory in the League Two play-off Final, he moved to Manchester
United in July 2012.

 Unable to break into the Manchester United first team, however, he went on loan to Wigan Athletic, Leicester City and Hull City before being released in June
2016.

 He joined Wigan Athletic in July 2016.

 He has represented England at under-16, under-17, under-18, under-19 and under-21 levels.
</item 6>

<item 7>
Trent Alexander-Arnold
Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.

 He is an academy graduate of Liverpool and made his senior debut for the club in 2016, aged 18.

 The following year, he won the club's Young Player of the Season award.

 He has also represented England at all youth levels from under-16 through to under-19.

 Due to the length of his name, his surname is shortened to Alexander for Premier League fixtures.
</item 7>

<item 8>
Gordon Burns: Scottish ex-player and Troon manager; played for Ayr. Jack Charlton: England’s 1966 World Cup-winning defender, Leeds legend and record
appearance-holder; Bobby’s brother. Barry Murphy: Irish international, one cap under Charlton. Adnan Januzaj: Belgian winger at Real Sociedad; former Manchester
United player and 2014 World Cup participant. Arthur Milton: England footballer and cricketer, with six Test caps. Leatherbarrow, McC
</item 8>

<item 9>
John Logan (footballer, born 1912)
John William Logan (16 August 1912 – October 1980) was an English professional footballer born in Horden, near Peterlee, County Durham who played in the
Football League for Darlington, Barnsley and Sheffield Wednesday.

 He played as a wing half.
</item 9>

<item 10>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 10>
=== END ITEM id=d8a3a5d2628d5486 ===

=== ITEM id=d8cccb75b760d888 ===
Gold sentences and facts:
Sentence 0: Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca O'Brien and Lauren Dark.
  Fact 0.0: Mediastan is a documentary film.
  Fact 0.1: Mediastan was released in 2013.
  Fact 0.2: Mediastan is about the 2010 United States diplomatic cables leak.
  Fact 0.3: Mediastan was directed by Johannes Wahlström.
  Fact 0.4: Mediastan was produced by Julian Assange.
  Fact 0.5: Mediastan was produced by Rebecca O'Brien.
  Fact 0.6: Mediastan was produced by Lauren Dark.
Sentence 1: Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he founded in 2006.
  Fact 1.0: Julian Paul Assange was born Julian Paul Hawkins.
  Fact 1.1: Julian Paul Assange was born on 3 July 1971.
  Fact 1.2: Julian Paul Assange is Australian.
  Fact 1.3: Julian Paul Assange is a computer programmer.
  Fact 1.4: Julian Paul Assange is the founder of WikiLeaks.
  Fact 1.5: WikiLeaks is an organisation.
  Fact 1.6: Julian Paul Assange founded WikiLeaks in 2006.

Retrieved items:
<item 1>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 1>

<item 2>
Aryanto Yuniawan
Aryanto Yuniawan (born June 18, 1977 in Magelang; age 40) is Chief Executive Officer of PT Mataram Surya Visi Sinema, a multimedia and animation movie company
located in Jogja Indonesia.

 Ary (his nickname) is best known as a director of Battle of Surabaya, the first 2D animation movie in Indonesia which got some awards i.e. Gold Remi Award USA
2016, GrandPrize Winner of SICAF 2016 in South Korea, Winner of Nice International Filmmaker Festival 2017 in France, Winner of Nioda International Film
Festival in India and others.

 He is also acclaimed as a writer, a producer, and currently working on several articles and books related to his job and hobby.

 He started his career from directing and producing independent animation films including music videos.

 He was trusted to be one of the juries of several film festivals such as Festival Film Indonesia (FFI 2016), Noida International Film Festival (NIFF 2016), and
Seoul International Cartoon and Animation Festival (SICAF 2017).
</item 2>

<item 3>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 3>

<item 4>
Shamal Sabri
Shamal Sabri (Kurdish: شه‌مال سه‌بری), (born 9 November 1985), is an independent Kurdish award-winning film producer and Duhok International Film Festival
Artistic Director, graduated from Duhok University with BA in English Literature.

.

 He was born in the city of Duhok (Kurdish: دهوك) in Iraqi Kurdistan.

 Apart from being one of the co-founders of Duhok IFF in 2009, he has worked in many other fields of video productions, produced many commercials and music
videos in NV Production and Vin TV.

 Throughout his film carrier, he has contributed in many award-winning films as an assistant director, production manager, assistant producer and line producer.

 His latest contributions were in many films such as "Before Snowfall" by Hisham Zaman, "The Swallow" by Mano Khalil, "Memories on Stone" by Shawkat Amin Korki
and Michael by Kurdo Duski.
</item 4>

<item 5>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 5>

<item 6>
Mediastan
Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca
O'Brien and Lauren Dark.
</item 6>

<item 7>
Shajoon Kariyal
He debuted as a director with "Rajaputhran" (1996), starring Suresh Gopi, Shobhana and Vikram.

 He has directed many films, including the commercially successful "Thachiledathu Chundan" (1999) and the critically acclaimed "Vadakkumnadhan" (2006).

 After "Vadakkumnadhan", he planned two films, "Raman Police" and "Talkies", but both the projects did not work out.

 In 2012, he directed "Chettayees" which he also co-produced, as one of the five partners of the newly launched production house Thakkaali Films.

 His latest film is "Sir C. P." (2015).
</item 7>

<item 8>
Shamal Sabri
He has proven himself over and over as an important player in Kurdistan's developing film industry and eventually earned the position of Artistic Director of
Duhok IFF in 2016.

 Shamal is the head of submission committee for films representing Iraq in the Asia Pacific Screen Awards.
</item 8>

<item 9>
Julian Assange
Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he
founded in 2006.

 He has won accolades including the Sam Adams Award and Martha Gellhorn Prize for Journalism.
</item 9>

<item 10>
Faraz Anwar (film)
Faraz Anwar – Documentary is a 2005 documentary film directed and produced by Haroon Sheikh.

 The documentary film follows the experience of Faraz Anwar, lead guitarist and vocalist of the Pakistani progressive rock band Mizraab, in the music industry.

 The documentary was released on November, 2005, by Poison Wood Productions.
</item 10>
=== END ITEM id=d8cccb75b760d888 ===

=== ITEM id=591473e9b9f2b6ba ===
Gold sentences and facts:
Sentence 0:  The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and Viv Albertine, who went on to play in The Slits.
  Fact 0.0: The band never played live.
  Fact 0.1: The band never released any recordings.
  Fact 0.2: Like London SS, the band is more famed for the number of band members who later became well known.
  Fact 0.3: Like Masters of the Backside, the band is more famed for the number of band members who later became well known.
  Fact 0.4: Sid Vicious is listed as a band member who later became well known as a member of the Sex Pistols.
  Fact 0.5: Keith Levene is listed as a band member who later became well known.
  Fact 0.6: Keith Levene was an early member of The Clash.
  Fact 0.7: Keith Levene was later a member of Public Image Ltd.
  Fact 0.8: Palmolive is listed as a band member who later became well known.
  Fact 0.9: Viv Albertine is listed as a band member who later became well known.
  Fact 0.10: Viv Albertine went on to play in The Slits.
Sentence 1:  Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
  Fact 1.0: The band was interviewed by a fanzine named SKUM despite never playing live.
  Fact 1.1: In the SKUM interview, Sid Vicious proclaimed 'I'll just be the yob that I am now'.
Sentence 2: Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock band the Sex Pistols.
  Fact 2.0: Sid Vicious was born John Simon Ritchie.
  Fact 2.1: Sid Vicious was born on 10 May 1957.
  Fact 2.2: Sid Vicious died on 2 February 1979.
  Fact 2.3: Sid Vicious was an English bassist.
  Fact 2.4: Sid Vicious was an English vocalist.
  Fact 2.5: Sid Vicious was most famous as a member of the influential punk rock band the Sex Pistols.

Retrieved items:
<item 1>
Sid Vicious
Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock
band the Sex Pistols.
</item 1>

<item 2>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 2>

<item 3>
The Idols with Sid Vicious
The Idols with Sid Vicious is a concert album of former Sex Pistols bassist Sid Vicious performing with The Idols; though recorded in September 1978, it wasn't
released until 1993.

 Steve Dior provided a recording of this performance to the New Rose record label.

 It was originally released by the Fan Club division of New Rose.

 It was also released in Japan by Teichiku Records, under license from New Rose.
</item 3>

<item 4>
Nancy Spungen
Nancy Laura Spungen (February 27, 1958 – October 12, 1978) was the American girlfriend of Sex Pistols bassist Sid Vicious and a figure of the 1970s punk rock
scene.

 Spungen's life and death have been the subject of controversy among music historians and fans of the Sex Pistols.
</item 4>

<item 5>
The Flowers of Romance (band)
The Flowers of Romance were an early punk band, formed in mid-1976 by Jo Faull and Sarah Hall.

 The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that
later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and
Viv Albertine, who went on to play in The Slits.

 Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
</item 5>

<item 6>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 6>

<item 7>
Glen Matlock
Glen Matlock (born 27 August 1956) is an English musician best known for being the bass guitarist in the original line-up of the punk rock band the Sex Pistols.

 He is sometimes credited as a co-author on 10 of the 12 songs on "Never Mind the Bollocks, Here's the Sex Pistols", although he had left the band by the time
the record was released and the bass was played by Steve Jones.

 He also continues to make his own records and tour with various bands, including the Sex Pistols.
</item 7>

<item 8>
Revolution in the Classroom
"Revolution in the Classroom" was a single released by the Ex Pistols; despite its packaging, the record has nothing to do with the Sex Pistols.

 The single was released in 1989 on the STP Records label, pressed on various colours of vinyl.

 The song, along with its B-side "Schools Are Prisons", were both included on the Ex-Pistols album, "Deny".

 They also appear disguised as genuine Sex Pistols recordings on two Sex Pistols retrospective albums by former producer Dave Goodman, "Pirates of Destiny" (I
Swirled Records, 1989) and "We Have Cum For Your Children" (Skyclad Records, 1988).

Under the title "Revolution" the song appears on a German Sex Pistols compilation "Anarchy in the USA" from 1992 released on MBC records.

 CD 084-60322 LP 008-60321
</item 8>

<item 9>
Bodies (Sex Pistols song)
"Bodies" is a Sex Pistols song about the shock of abortion from the 1977 album "Never Mind the Bollocks, Here's the Sex Pistols".

 The lyrics contain very graphic imagery about a terminated fetus and feature a great deal of profanity for the time: the third and final verse begins with a
couplet in which the word "fuck" is repeated five times in rapid succession.

 Along with the later "Belsen Was a Gas," it is probably the most graphic and controversial Sex Pistols song in both its subject matter and style.

 Musically, it is also the fastest and heaviest song in the Sex Pistols canon — characterized by thudding drums, droning buzzsaw guitar, and shouted vocals.

 As such, it can be considered a significant antecedent to the genres of hardcore, thrash, and speed metal that was to emerge in the 1980s.
</item 9>

<item 10>
Ex Pistols
The Ex Pistols were a scam act introduced in 1979 by one-time Sex Pistols producer Dave Goodman after his services were substituted for those of professional
producers.

 The Ex Pistols existed as a sound-alike band meant to mislead fans due to the similar music, name, and artwork, that led people to believe their songs were
actually new or lost Sex Pistols recordings.
</item 10>
=== END ITEM id=591473e9b9f2b6ba ===

=== ITEM id=dfeb059f6e8728f4 ===
Gold sentences and facts:
Sentence 0: Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.
  Fact 0.0: Margaret Landon was born September 7, 1903.
  Fact 0.1: Margaret Landon died December 4, 1993.
  Fact 0.2: Margaret Landon was an American writer.
  Fact 0.3: Margaret Landon is best remembered for 'Anna and the King of Siam'.
  Fact 0.4: 'Anna and the King of Siam' is Margaret Landon's best-selling novel, published in 1944.
  Fact 0.5: 'Anna and the King of Siam' is a novel about the life of Anna Leonowens.
  Fact 0.6: 'Anna and the King of Siam' eventually sold over a million copies.
  Fact 0.7: 'Anna and the King of Siam' was translated into more than twenty languages.
Sentence 1:  In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.
  Fact 1.0: In 1950, Margaret Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II.
  Fact 1.1: Richard Rodgers and Oscar Hammerstein II created the musical 'The King and I' from Margaret Landon's book.
Sentence 2:  A later work, "Never Dies the Dream", appeared in 1949.
  Fact 2.0: 'Never Dies the Dream' is a later work.
  Fact 2.1: 'Never Dies the Dream' appeared in 1949.
Sentence 3: Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.
  Fact 3.0: Dame Elizabeth Rosemond Taylor, DBE, was born February 27, 1932.
  Fact 3.1: Dame Elizabeth Rosemond Taylor, DBE, died March 23, 2011.
  Fact 3.2: Dame Elizabeth Rosemond Taylor was a British-American actress.
  Fact 3.3: Dame Elizabeth Rosemond Taylor was a businesswoman.
  Fact 3.4: Dame Elizabeth Rosemond Taylor was a humanitarian.
Sentence 4:  She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.
  Fact 4.0: Elizabeth Taylor began her career as a child actress in the early 1940s.
  Fact 4.1: Elizabeth Taylor was one of the most popular stars of classical Hollywood cinema in the 1950s.
Sentence 5:  She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.
  Fact 5.0: Elizabeth Taylor continued her career successfully into the 1960s.
  Fact 5.1: Elizabeth Taylor remained a well known public figure for the rest of her life.
Sentence 6:  In 1999, the American Film Institute named her the seventh-greatest female screen legend.
  Fact 6.0: In 1999, the American Film Institute named Elizabeth Taylor the seventh-greatest female screen legend.

Retrieved items:
<item 1>
Mary Elizabeth Bliss
Mary Elizabeth "Betty" Taylor Bliss Dandridge, born Mary Elizabeth Taylor (April 20, 1824 – July 25, 1909), was the youngest of the three surviving daughters of
President Zachary Taylor (1849-1850) and Margaret Mackall Smith Taylor.
</item 1>

<item 2>
Anna and the King of Siam (novel)
Anna and the King is a 1944 semi-fictionalized biographical novel by Margaret Landon.
</item 2>

<item 3>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 3>

<item 4>
Krupp Diamond
The Elizabeth Taylor Diamond, formerly known as the Krupp Diamond, is a 33.19 carat diamond that was bought by Richard Burton for his wife, Elizabeth Taylor in
1968.

 The Krupp diamond was one of a number of significant pieces of jewellery owned by Taylor, her collection also included the 68 carat Taylor–Burton Diamond,
which was bought by the couple in 1969.

 The Krupp diamond was sold by Taylor's estate in 2011 for $8.8 million.
</item 4>

<item 5>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 5>

<item 6>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 6>

<item 7>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 7>

<item 8>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 8>

<item 9>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 9>

<item 10>
Margaret Landon
Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of
the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.

 In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.

 A later work, "Never Dies the Dream", appeared in 1949.
</item 10>
=== END ITEM id=dfeb059f6e8728f4 ===

=== ITEM id=e0cf7facbb5af4bd ===
Gold sentences and facts:
Sentence 0: Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.
  Fact 0.0: Catfish and the Bottlemen are a British rock band.
  Fact 0.1: Catfish and the Bottlemen were formed in Llandudno, Wales.
  Fact 0.2: Catfish and the Bottlemen were formed in 2007.
Sentence 1: The Lumineers are an American folk rock/Americana band based in Denver, Colorado.
  Fact 1.0: The Lumineers are an American folk rock/Americana band.
  Fact 1.1: The Lumineers are based in Denver, Colorado.
Sentence 2:  Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.
  Fact 2.0: Schultz and Fraites began writing together in 2005.
  Fact 2.1: Schultz and Fraites began performing together in 2005.
  Fact 2.2: Schultz and Fraites began writing together in Ramsey, New Jersey.
  Fact 2.3: Schultz and Fraites began performing together in Ramsey, New Jersey.
Sentence 3:  The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.
  Fact 3.0: The Lumineers emerged as one of the most popular folk rock/Americana artists.
  Fact 3.1: The Lumineers emerged during the revival of folk rock and Americana genres.
  Fact 3.2: The Lumineers emerged during the growing popularity of folk rock and Americana in the 2010s.

Retrieved items:
<item 1>
The Balcony (album)
The Balcony is the debut studio album by Welsh rock band Catfish and the Bottlemen.

 It was released on 15 September 2014 in the United Kingdom and 15 January 2015 in the United States.

 On 20 March 2015, "The Balcony" was awarded a Gold certification from the British Phonographic Industry.

 The album cover depicts outlines of two headless people pleasuring each other's genitals.

 The illustration is by New York artist Tim Lahan who originally posted it to his Flickr account in 2009 and was subsequently contacted by the band.
</item 1>

<item 2>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 2>

<item 3>
The Ride (Catfish and the Bottlemen album)
The Ride is the second studio album by British indie rock band Catfish and the Bottlemen.

 It was released on 27 May 2016 through Capitol Records.
</item 3>

<item 4>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 4>

<item 5>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 5>

<item 6>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 6>

<item 7>
Billy Bibby &amp; The Wry Smiles
Billy Bibby & The Wry Smiles are a British indie rock band formed in the autumn of 2015 in the area around Llandudno, North Wales and Chester, UK.

 The band is fronted by Billy Bibby, former lead guitarist and founding member of Catfish and the Bottlemen.

 Bibby parted ways with Catfish and the Bottlemen in the summer of 2014, shortly after co-writing and recording their Platinum selling debut album, "The
Balcony".

 Soon thereafter, Billy Bibby began writing songs and playing acoustic shows in various UK cities in order to test public response to his new music.

 In the spring of 2015, Billy Bibby began seeking musicians to form a new band.

 Auditions were conducted with the help of Simon Jones, formerly of The Verve.

 Band members Rob Jones (lead guitar), Matt Bassy (bass), and Michael Pearce (drums) were selected to form the four-piece ensemble.
</item 7>

<item 8>
The Ride Tour
The Ride Tour is the ongoing second concert tour by English indie rock band Catfish and the Bottlemen, in support of the group's second studio album "The Ride".

 The tour was announced in February 2016 and began on 4 April 2016, in Glasgow, United Kingdom at the O2 Academy Glasgow.

 The tour is planned to travel across Europe and North America with a total of 50 shows planned so far.

 On 22 February 2016, the band announced an April 2016 stint of shows in UK, including Glasgow, Halifax, Southend On Sea, Doncaster, London and Brighton.

 The band will also headline several festivals across the UK throughout the summer, including Liverpool's 'Sound City' in May 2016, Bristol's Summer Series in
July 2016 and Manchester's Castlebowl in July 2016.

 On 14 March 2016 it was announced that the band will play several dates in North America in June 2016.
</item 8>

<item 9>
Soundcheck (song)
"Soundcheck" is the eighth single by Welsh indie rock band, Catfish and the Bottlemen.

 The song was the lead single off of their sophomore album, "The Ride".

 The single was a promotional single that was digitally released on 16 February 2016.
</item 9>

<item 10>
Bloc Party
Bloc Party are an English indie rock band, currently composed of Kele Okereke (lead vocals, rhythm guitar, keyboards, sampler), Russell Lissack (lead guitar,
keyboards), Justin Harris (bass guitar, keyboards, saxophones, backing vocals) and Louise Bartle (drums, percussion).

 Former members Matt Tong and Gordon Moakes left the band in 2013 and 2015 respectively.

 Their brand of music, whilst rooted in rock, retains elements of other genres such as electronica and house music.

 The band was formed at the 1999 Reading Festival by Okereke and Lissack.

 They went through a variety of names before settling on Bloc Party in 2003.

 Moakes joined the band after answering an advert in "NME" magazine, while Tong was picked via an audition.

 Bloc Party got their break by giving BBC Radio 1 DJ Steve Lamacq and Franz Ferdinand's lead singer, Alex Kapranos, a copy of their demo "She's Hearing Voices".
</item 10>
=== END ITEM id=e0cf7facbb5af4bd ===

=== ITEM id=34890fc76c8684c5 ===
Gold sentences and facts:
Sentence 0:  The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
  Fact 0.0: The show originally ran from 7 September 1977 to 20 November 1995.
  Fact 0.1: The show was hosted by Gordon Burns.
  Fact 0.2: The show was usually broadcast on the ITV network.
  Fact 0.3: The show was usually broadcast on Mondays at 7pm.
Sentence 1:  He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West Tonight" from January 1997 to October 2011.
  Fact 1.0: The person referred to as 'He' was the host of 'The Krypton Factor' for its original 18-year run.
  Fact 1.1: The Krypton Factor's original run was 1977–1995.
  Fact 1.2: The person referred to as 'He' was the chief anchorman for the BBC regional news programme 'North West Tonight'.
  Fact 1.3: The person referred to as 'He' was the chief anchorman for 'North West Tonight' from January 1997 to October 2011.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
The Krypton Factor
The Krypton Factor is a British game show produced by Granada for broadcast on ITV.

 The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
</item 2>

<item 3>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 3>

<item 4>
At Your Request
At Your Request was an Australian television daytime series which aired from 1958 to 1959.

 The series aired on Tuesdays at 2:30PM on Melbourne station HSV-7, and was hosted by baritone Charles Skase, who was also known as a radio personality.

 Information on this series is scarce.

 The series presented requested songs, but it is not clear how these songs were presented (such as whether it was a disc jockey series like "TV Disc Jockey", a
lip-sync series like "Hit Parade", or a live music series like "Sweet and Low").

 The archival status of the series is also not known, although being a daytime series aired in a single city means it is unlikely (though not impossible) that
kinescope recordings exist of it.
</item 4>

<item 5>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 5>

<item 6>
Gordon Burns (footballer)
Gordon Burns (born 2 December 1978) is a Scottish former professional footballer who is currently manager of Troon in the Scottish Junior Football Association,
West Region.

 He has previously played in the Scottish Championship for Ayr United.
</item 6>

<item 7>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 7>

<item 8>
Gordon Burns
Gordon Henry Burns (born 10 June 1942) is a Northern Irish journalist and broadcaster.

 He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West
Tonight" from January 1997 to October 2011.

 In November 2011 Gordon moved back to Belfast were he was born.
</item 8>

<item 9>
Night After Night with Allan Havey
Night After Night with Allan Havey was a late night talk show which aired on The Comedy Channel and, later, Comedy Central from 1989 to 1992.

 The show featured the stream of consciousness of host Allan Havey, and other departures from the standard late-night talk show format.

 Havey's sidekick was Nick Bakay, who moved on to act as sidekick on Dennis Miller's short-lived late night talk show.
</item 9>

<item 10>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 10>
=== END ITEM id=34890fc76c8684c5 ===

=== ITEM id=ce85f09f4d91d518 ===
Gold sentences and facts:
Sentence 0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great Britain and is dedicated to Saint Chad of Mercia.
  Fact 0.0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham.
  Fact 0.1: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the province of the Catholic Church in Great Britain.
  Fact 0.2: The Metropolitan Cathedral Church and Basilica of Saint Chad is dedicated to Saint Chad of Mercia.
Sentence 1:  Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English Reformation and raised to cathedral status in 1852.
  Fact 1.0: St Chad's was built by Augustus Welby Pugin.
  Fact 1.1: St Chad's was substantially complete by 1841.
  Fact 1.2: St Chad's is one of the first four Catholic churches constructed after the English Reformation.
  Fact 1.3: St Chad's was raised to cathedral status in 1852.
Sentence 2: Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.
  Fact 2.0: Chelmsford Cathedral is in the city of Chelmsford, Essex, England.
  Fact 2.1: Chelmsford Cathedral is dedicated to St Mary the Virgin.
  Fact 2.2: Chelmsford Cathedral is dedicated to St Peter.
  Fact 2.3: Chelmsford Cathedral is dedicated to St Cedd.
Sentence 3:  It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
  Fact 3.0: Chelmsford Cathedral became a cathedral when the Anglican Diocese of Chelmsford was created in 1914.
  Fact 3.1: Chelmsford Cathedral is the seat of the Bishop of Chelmsford.

Retrieved items:
<item 1>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 1>

<item 2>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
</item 2>

<item 3>
St Chad's College, Durham
St Chad's College is a recognised (independent) college of Durham University in England, founded in 1904 as an Anglican hall for the training of Church of
England clergy.

 The main part of the college is located on the Bailey, occupying nine historic buildings at the east end of Durham Cathedral.

 It neighbours Hatfield College to its north, while St John's College and St Cuthbert's Society are to its south.

 The college is named after St Chad of Mercia, a 7th-century bishop.
</item 3>

<item 4>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 4>

<item 5>
Dean of Chelmsford
The Dean of Chelmsford is the head ("primus inter pares" – first among equals) and chair of the Cathedral Chapter, the governing body of Chelmsford Cathedral,
the Cathedral Church of St Mary the Virgin, St Peter and St Cedd.

 Before 2000 the post was designated as a provost, which was then the equivalent of a dean at most English cathedrals.

 The cathedral is the mother church of the Diocese of Chelmsford and seat of the Bishop of Chelmsford.

 The Dean of Chelmsford is also responsible for the Chapel of St Peter-on-the-Wall at Bradwell-on-Sea, founded by St Cedd, among the oldest church buildings in
regular use in England.
</item 5>

<item 6>
Guildford Cathedral, designed by Edward Maufe and built 1936–1961 on land donated by Lord Onslow, is the Church of England Diocese of Guildford’s seat. Its
exterior features sculpture by notable mid-century artists. Dean Dianna Gwilliams leads its chapter; Lanesborough School is its choir school. Barry Rose
established its choir and daily sung worship, while organist Katherine Dienes is the first woman to hold a Church of England cathedral’s senior musical post.
The
</item 6>

<item 7>
St Philip's Cathedral, Birmingham
The Cathedral Church of Saint Philip is the Church of England cathedral and the seat of the Bishop of Birmingham.

 Built as a parish church and consecrated in 1715, St Philip's became the cathedral of the newly formed Diocese of Birmingham in 1905.

 St Philip's was built in the early 18th century in the Baroque style by Thomas Archer and is located on Colmore Row, Birmingham, England.

 The cathedral is a Grade I listed building.

 St Philip's is the third smallest cathedral in England after Derby and Chelmsford.
</item 7>

<item 8>
The entries cover British and Australian church heritage. London’s St Paul’s, founded on the site of a church dating to 604, is Wren’s Grade I-listed Baroque
cathedral, rebuilt after the Great Fire. Rockhampton’s St Paul’s Cathedral (1883–1953) and adjacent hall (built circa 1900) are Queensland-listed. Augustine
became Canterbury’s first archbishop in 597. Other cathedrals include medieval Lichfield (England’s only
</item 8>

<item 9>
St Chad's Cathedral, Birmingham
The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great
Britain and is dedicated to Saint Chad of Mercia.

 Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English
Reformation and raised to cathedral status in 1852.

 It is one of only four minor basilicas in England (the others being Downside Abbey, the National Shrine of Our Lady at Walsingham and Corpus Christi Priory,
this last now disused).

 St Chad's is a Grade II* listed building.

 The cathedral is located in a public greenspace near St Chad's Queensway, in central Birmingham.

 s of 2014 the Archbishop was Bernard Longley and the Dean Canon Gerry Breen.
</item 9>

<item 10>
Bishop's House, Birmingham
The Bishop's House in Birmingham, England was designed by Augustus Pugin as the residence of Thomas Walsh, the first Roman Catholic Bishop of Birmingham.

 It was situated opposite St Chad's Cathedral, on the corner of Bath Street and Weaman Street in Birmingham City Centre.
</item 10>
=== END ITEM id=ce85f09f4d91d518 ===

=== ITEM id=5ca286f65125ed9f ===
Gold sentences and facts:
Sentence 0: Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children taking up residence in a small town in Maine.
  Fact 0.0: Summer Magic is a 1963 Walt Disney Productions film.
  Fact 0.1: Summer Magic stars Hayley Mills.
  Fact 0.2: Summer Magic stars Burl Ives.
  Fact 0.3: Summer Magic stars Dorothy McGuire.
  Fact 0.4: Summer Magic is a story about a Boston widow and her children.
  Fact 0.5: In Summer Magic, the Boston widow and her children take up residence in a small town in Maine.
Sentence 1: Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.
  Fact 1.0: Around the World in 80 Days is a 2004 American film.
  Fact 1.1: Around the World in 80 Days is an action-adventure comedy family film.
  Fact 1.2: Around the World in 80 Days is based on Jules Verne's novel of the same name.

Retrieved items:
<item 1>
Around the World in 80 Days (2004 film)
Around the World in 80 Days is a 2004 American action-adventure comedy family film based on Jules Verne's novel of the same name.

 It stars Jackie Chan, Steve Coogan and Cécile de France.

 The film is set in 19th-century Britain and centers on Phileas Fogg (Steve Coogan), here reimagined as an eccentric inventor, and his efforts to circumnavigate
the globe in 80 days.

 During the trip, he is accompanied by his Chinese valet, Passepartout (Jackie Chan).

 For comedic reasons, the film intentionally deviated wildly from the novel and included a number of anachronistic elements.

 With production costs of about $110 million and estimated marketing costs of $30 million, it earned $24 million at the U.S. box office and $72 million
worldwide, making it a box office flop.

 It was Arnold Schwarzenegger's last film before he took a hiatus from acting to become Governor of California until 2010's "The Expendables".
</item 1>

<item 2>
Summer Holiday (1963 film)
Summer Holiday is a British CinemaScope and Technicolor musical film featuring singer Cliff Richard.

 The film was directed by Peter Yates (his debut), produced by Kenneth Harper.

 The original screenplay was written by Peter Myers and Ronald Cass (who also wrote most of the song numbers and lyrics).

 The cast includes Lauri Peters, Melvyn Hayes, Teddy Green, Jeremy Bulloch, Una Stubbs, Pamela Hart, Jacqueline Daryl, Lionel Murton, Madge Ryan, David Kossoff,
Nicholas Phipps, Ron Moody and The Shadows.

 Herbert Ross choreographed the musical numbers.
</item 2>

<item 3>
Around the World in Eighty Days
Around the World in Eighty Days (French: "Le tour du monde en quatre-vingts jours" ) is a classic adventure novel by the French writer Jules Verne, published in
1873.

 In the story, Phileas Fogg of London and his newly employed French valet Passepartout attempt to circumnavigate the world in 80 days on a £20,000 wager (the
approximate equivalent of £2 million in 2016) set by his friends at the Reform Club.

 It is one of Verne's most acclaimed works .
</item 3>

<item 4>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 4>

<item 5>
Around the World in 80 Days (1956 film)
Around the World in 80 Days (sometimes spelled as Around the World in Eighty Days) is a 1956 American epic adventure-comedy film starring Cantinflas and David
Niven , produced by the Michael Todd Company and released by United Artists.
</item 5>

<item 6>
A Boy Scout Around the World
A Boy Scout Around the World (Danish: "Jorden Rundt i 44 dage", literally: "Around the World in 44 Days") is a travel description published in October 1928 and
written by Danish Boy Scout and later actor Palle Huld at the age of 15 following his travel around the world in spring 1928.

 His trip was sponsored by a Danish newspaper and made on the occasion of the 100 birthday of Jules Verne a French author of adventure and science fiction.

 Palle Huld was chosen after having answered to an ad in the newspaper; applicants had to be boys, 15 years old, able to manage in English and German and of
good health.

 Like the characters in Jules Verne’s novel "Around the World in 80 days" he was only allowed to travel by land and sea, not by air.

 The travel (on first class) went from Denmark to Great Britain, across the Atlantic Ocean to Canada.

 From the American west coast he continued to Japan, China, Soviet Union, Poland, Germany and back to Denmark.
</item 6>

<item 7>
Around the World (1956 song)
"Around the World" was the theme tune from the 1956 movie "Around the World in 80 Days" In the film, only an instrumental version of the song appeared, although
the vocal version has become by far the better known one.

 The song was written by Harold Adamson and Victor Young; Young died in 1956, several weeks after the film's release and he received the Academy Award for Best
Music, Scoring of a Dramatic or Comedy Picture posthumously.

 Young's orchestral version was a #13 hit on the Billboard charts in 1957.

 The recording by Bing Crosby was the B-side of the Victor Young version in 1957, on Festival SP45-1274 in Australia, and was a joint charting success.
</item 7>

<item 8>
Around the World in 80 Days (1988 film)
Around the World in 80 Days is an Australian 48-minute direct-to-video animated film from Burbank Films Australia.

 It was originally released in 1988.

 The film is based on Jules Verne's classic French novel, "Around the World in 80 Days", first published in 1873, and was adapted by Leonard Lee.

 It was produced by Roz Phillips and featured original music by Simon Walker.

 The film imitated BRB Internacional's Spanish 1981 series, "La vuelta al mundo de Willy Fog", in its use of anthropomorphic animals in the human roles.

 The copyright in this film is now owned by Pulse Distribution and Entertainment and administered by digital rights management firm NuTech Digital.
</item 8>

<item 9>
Summer Magic
Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children
taking up residence in a small town in Maine.

 The film was based on the novel "Mother Carey's Chickens" by Kate Douglas Wiggin and was directed by James Neilson.

 This was the fourth of six film Mills did for Disney, and the young actress received a Golden Globe nomination for her work here.
</item 9>

<item 10>
The entries cover Jules Verne’s 1873 novel, in which Phileas Fogg wagers £20,000 he can circle the globe in 80 days, and its adaptations: 1956 and 2004 films, a
1988 animation, and Michael Palin’s 1989 BBC series. The 2004 film flopped; the 1956 film’s theme became a hit. Danish Scout Palle Huld’s 1928, 44-day, no
</item 10>
=== END ITEM id=5ca286f65125ed9f ===

=== ITEM id=9ec898b568bfcdc7 ===
Gold sentences and facts:
Sentence 0: Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.
  Fact 0.0: Ash Lieb was born on 22 August 1982.
  Fact 0.1: Ash Lieb is an Australian artist.
  Fact 0.2: Ash Lieb is an Australian writer.
  Fact 0.3: Ash Lieb is an Australian comedian.
  Fact 0.4: Ash Lieb is known for his surreal humour.
  Fact 0.5: Ash Lieb is known for his surreal art.
Sentence 1: Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.
  Fact 1.0: Helen Elswyth Thane Ricker Beebe was born on May 16, 1900.
  Fact 1.1: Helen Elswyth Thane Ricker Beebe died on July 31, 1984.
  Fact 1.2: Helen Elswyth Thane Ricker Beebe was an American romance novelist.

Retrieved items:
<item 1>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 1>

<item 2>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 2>

<item 3>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 3>

<item 4>
Tryst (novel)
Tryst, written in 1939 by Elswyth Thane, is a story of two people and a seemingly ordinary home.

 While a quick summary may make it sound like a Horror novel, it actually borders on Mystery and Romance.
</item 4>

<item 5>
Ash Lieb
Ash Lieb (born 22 August 1982) is an Australian artist, writer and comedian, known for his surreal humour and art.

 Born in Ballarat, Ash Lieb began exhibiting art at eight years of age, and at the age of fifteen, wrote his first novel, "The Secret Well".

 Throughout his career, Lieb has created a diverse range of artworks, books, short films, and comedic performances, which have often possessed philosophical or
psychiatric undertones.
</item 5>

<item 6>
Elswyth Thane
Helen Elswyth Thane Ricker Beebe (May 16, 1900 – July 31, 1984) was an American romance novelist.

 Born in Burlington, Iowa, she was the daughter of a local teacher and high school principal.

 The family moved to New York City in 1918, and "Helen Ricker" changed her name to "Elswyth Thane".

 She began working as a freelance writer in the 20s, and became a newspaper writer and a Hollywood screenwriter.

 Her first novel, "Riders of the Wind", was published in 1926.

 Her novel, "The Tudor Wench", about Elizabeth I of England, was made into a play.

 She was a collector of scarves.
</item 6>

<item 7>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 7>

<item 8>
Hans Aschenborn
Hans Anton Aschenborn (1 February 1888 – 10 April 1931) was a renowned animal painter of African wildlife.

 He worked both in Germany and in southern Africa.

 His work is featured in the older German Thieme-Becker or Saur art encyclopedia.

 A Master of Arts thesis by Karin Skawran concerning the graphic works of Hans Anton Aschenborn was published in the South African art and culture periodical,
"Lantern" in 1965.)

 In 1963 the University of Pretoria published a book about Aschenborn as an artist entitled, "Hans Anton Aschenborn : Mens en Kunstenaar".

 In 1970 another one followed by the Pretoria Art Museum (South Africa).

 Other publications on Aschenborn feature his etchings and linocuts.
</item 8>

<item 9>
Lucille Watahomigie
Thew AILDI was the inspiration for the Canadian Indigenous Languages and Literacy Development Institute (CILLDE) which was formed in 1999.

 She has also taught at the institute since its inception and continues to teach there on a regular basis.

 Some of her other published works include "Spirit Mountain: A Yuman Anthology", of which she was an editor, as well as works on bilingual education,
ethnobotany, education, linguistics and language revitalization.
</item 9>

<item 10>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 10>
=== END ITEM id=9ec898b568bfcdc7 ===

=== ITEM id=ce32ea058e6f5774 ===
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
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 1>

<item 2>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 2>

<item 3>
First North Americans
The novels take into account new developments in North American archaeology such as the discovery of Kennewick Man and the development of the coastal-route
model as a possible alternative or supplement to overland migration across Beringia.
</item 3>

<item 4>
Kate Atkinson’s 2015 *A God in Ruins*, companion to *Life After Life*, follows Teddy Todd in WWII and won the Costa Novel Award. Other works include John
Howard’s Central/Eastern European fiction; Conrad’s *Heart of Darkness*, Marlow’s Congo voyage and Kurtz obsession; Keri Hulme’s Booker-winning *The Bone
People*; Karen Joy Fowler’s PEN/Faulkner-winning, Booker-shortlisted *We Are All Completely Beside Ours
</item 4>

<item 5>
The Stranger (novel)
L’Étranger (The Outsider [UK], or The Stranger [US]) is a 1942 novel by French author Albert Camus.

 Its theme and outlook are often cited as examples of Camus' philosophy of the absurd and existentialism, though Camus personally rejected the latter label.
</item 5>

<item 6>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 6>

<item 7>
Season of Migration to the North
Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist
Tayeb Salih.

 Originally published in Arabic in 1966, it has since been translated into more than twenty languages.

 Salih was fluent in both English and Arabic, but chose to pen this novel in Arabic.

 The English translation was published in 1969 as part of the influential Heinemann African Writers Series.

 The novel is a counter narrative to "Heart of Darkness".

 It was described by Edward Said as one of the six great novels in Arabic literature.

 In 2001 it was selected by a panel of Arab writers and critics as the most important Arab novel of the twentieth century.
</item 7>

<item 8>
The entries profile writers, scholars, artists, publishers, and advocates. Highlights include computer scientist Nuria Oliver’s data-science leadership and
human-behavior research; Helen Keller’s childhood story “The Frost King”; Booker Prize winner Keri Hulme; Alice Walker’s Pulitzer-winning *The Color Purple* and
anti-FGM work; and Lucille Watahomigie’s Hualapai language revitalization. Others include novelists Sarra Manning and Kim Edwards, poet Tara Betts
</item 8>

<item 9>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=ce32ea058e6f5774 ===

=== ITEM id=cde7c48cdecec199 ===
Gold sentences and facts:
Sentence 0: The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
  Fact 0.0: The International Grenfell Association (IGA) is an organization.
  Fact 0.1: The International Grenfell Association was founded by Sir Wilfred Grenfell.
  Fact 0.2: The International Grenfell Association provides health care.
  Fact 0.3: The International Grenfell Association provides education.
  Fact 0.4: The International Grenfell Association provides religious services.
  Fact 0.5: The International Grenfell Association provides rehabilitation.
  Fact 0.6: The International Grenfell Association provides other social activities.
  Fact 0.7: The International Grenfell Association serves fisherman and coastal communities in northern Newfoundland.
  Fact 0.8: The International Grenfell Association serves fisherman and coastal communities on the coast of Labrador.
Sentence 1: Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
  Fact 1.0: Sir Wilfred Thomason Grenfell held the title KCMG.
  Fact 1.1: Sir Wilfred Thomason Grenfell was born on 28 February 1865.
  Fact 1.2: Sir Wilfred Thomason Grenfell died on 9 October 1940.
  Fact 1.3: Sir Wilfred Thomason Grenfell was a medical missionary to Newfoundland.
  Fact 1.4: Sir Wilfred Thomason Grenfell was a medical missionary to Labrador.

Retrieved items:
<item 1>
Church of St. Ignatius Loyola (New York City)
The Church of St. Ignatius of Loyola is a Roman Catholic parish church located on the Upper East Side of Manhattan, New York City, administered by the Society
of Jesus (Jesuits).

 The parish is under the authority of the Archdiocese of New York, and was established in 1851 as St. Lawrence O'Toole's Church.

 In 1898, permission to change the patron saint of the parish from St. Lawrence O’Toole to St. Ignatius of Loyola was granted by Rome.

 The address is 980 Park Avenue, New York City, New York 10028.

 The church on the southwest corner of Park Avenue and 84th Street is part of a Jesuit complex on the block that includes Wallace Hall, the parish hall, beneath
the church, the rectory at the midblock location on Park Avenue, the grade school of St.
</item 1>

<item 2>
Wilfred Grenfell
Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
</item 2>

<item 3>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 3>

<item 4>
David Jackman (politician)
David Ignatius "Nish" Jackman (May 19, 1902 – March 23, 1967) was a labour leader and politician in Newfoundland.

 He represented Harbour Main-Bell Island in the Newfoundland House of Assembly from 1949 to 1956.
</item 4>

<item 5>
International Grenfell Association
The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and
rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
</item 5>

<item 6>
Angelino Apelar
Angelino "Andy" Baldemor Apelar (June 16, 1927 - January 23, 2006) was an Evangelical Christian leader in the Filipino American community, and a founder of the
Association of Filipino Churches (AFC) of the Christian and Missionary Alliance.

 He had served as a missionary/pastor and radio broadcaster in the Philippines before moving to the United States to start the AFC.

 He served as president for several terms.

 He graduated from the Far Eastern Bible Institute and Seminary, where he also met his future wife, Purita Palomar, a member of one of Capiz province's
influential families.

 They have two children: Faith and Jemuel Apelar.
</item 6>

<item 7>
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
</item 7>

<item 8>
Training centre for release of the Atma-energy
Training centre for release of the Atma-energy (German: "Trainingszentrum zur Freisetzung der Atmaenergie"), also known as "Atman Foundation", was a new
religious movement active mainly on the island of Tenerife and in Germany.

 This sect was originally a splinter group of the Brahma Kumaris and is known for a police and media scare in which an alleged attempt to commit ritual suicide
took place in Teide National Park in Tenerife.

 The group believed in the end of the world but according to the religious studies scholar Georg Schmid and the sociologist Massimo Introvigne had no intention
of collective suicide.
</item 8>

<item 9>
Mary's Harbour
Mary's Harbour is a town in the Canadian province of Newfoundland and Labrador.

 The town had a population of 474 in the Canada 2006 Census.

 It is serviced by Mary's Harbour Airport.

 Mary's Harbour surrounds the St. Mary's River.

 St. Mary's River was the site of a salmon fishery as early as the 1780s.

 However Mary's Harbour was not a permanent settlement until after a fire at Battle Harbour in 1930.

 The International Grenfell Association decided to relocate its hospital and boarding school, destroyed by the fire, from Battle Harbour to Mary's Harbour.

 Mary's Harbour has always depended on the fishery for its livelihood.

 Since the Collapse of the Atlantic northwest cod fishery the community has thrived on the crab fishery.

 The Labrador Fishermen's Union Shrimp Company employs over 120 people at the local crab processing facility.

 It is also the gateway to the National Historic District of Battle Harbour.
</item 9>

<item 10>
Ellis Gowing
Gowing was educated at Fort Street High School and the University of Sydney; and ordained in 1907.

 He began his career as a Curate at The Oaks, New South Wales, after which he served at Armidale Cathedral.

 Coming to England he was on the staff of St James the Less, Bethnal Green then domestic chaplain to the Bishop of Chelmsford from 1914 to 1917.

 He was vicar of Prittlewell from 1917 until his death; an honorary canon of Chelmsford Cathedral from 1921 to 1938 and Rural Dean of Canewdon from 1918 to
1938.

 He was given the Freedom of the County Borough of Southend-on-Sea in 1953.
</item 10>
=== END ITEM id=cde7c48cdecec199 ===
