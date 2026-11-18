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

This file has 12 items.

=== ITEM id=4b05b5781306d08e ===
Gold sentences and facts:
Sentence 0:  The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
  Fact 0.0: The film concentrated on Ellsworth "Bumpy" Johnson.
  Fact 0.1: Laurence Fishburne played Ellsworth "Bumpy" Johnson in the film.
  Fact 0.2: The film concentrated on Dutch Schultz.
  Fact 0.3: Tim Roth played Dutch Schultz in the film.
  Fact 0.4: The film concentrated on Lucky Luciano.
  Fact 0.5: Andy García played Lucky Luciano in the film.
Sentence 1:  Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.
  Fact 1.0: Lucky Luciano is considered the father of modern organized crime in the United States.
  Fact 1.1: Lucky Luciano is considered the father of modern organized crime in the United States for establishing the first Commission.
  Fact 1.2: Lucky Luciano established the first Commission.

Retrieved items:
<item 1>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 1>

<item 2>
Frank Colacurcio
Francis Colacurcio, Sr. (June 18, 1917 – July 2, 2010) was an American businessman and boss of the Seattle crime family known for running strip clubs in
Seattle, Washington.

 He gained notoriety as a subject of ongoing federal investigations into organized crime in the city and was suspected of being an organized crime boss.
</item 2>

<item 3>
James Marcello
James J. Marcello (born December 13, 1943), also known variously as Little Jimmy, Jimmy Light and as Jimmy the Man Marcello, is an imprisoned crime boss who was
a front boss for the Chicago Outfit criminal organization in the 1980s, 1990s and early 2000s.

 Organized crime observers identified Marcello as a figurehead during that period while the organization's day-to-day operations actually were run by John “No
Nose” DiFronzo, Joseph “Joey the Clown” Lombardo, Joseph “Joe the Builder” Andriacchi and Angelo J. LaPietra.

 Marcello is the Cousin of Michael Hayes Marcello, and a cousin of Robert Urbinatti.

 He is also the father of Rocco Marcello.
</item 3>

<item 4>
Hoodlum (film)
Hoodlum is a 1997 American crime drama film that gives a fictionalized account of the gang war between the Italian/Jewish mafia alliance and the Black gangsters
of Harlem that took place in the late 1920s and early 1930s.

 The film concentrated on Ellsworth "Bumpy" Johnson (Laurence Fishburne), Dutch Schultz (Tim Roth), and Lucky Luciano (Andy García).
</item 4>

<item 5>
Illtown
Dante and his girlfriend Micky run a very profitable drug operation in a seaside town, aided and abetted by a host of teens who sell the smack at discos around
town, as well as by Lucas, a corrupt cop who's on the take.

 Their downfall comes when they suspect one of the boys, Pep, of ripping them off, and his accidental death causes disloyalty among the teens, who suspect Dante
offed them.

 All of this is perfect for the return of Gabriel, a one-time partner of Dante, who has just been released from jail, and has an almost angelic demeanor and the
certainty that he can fix everyone's lives.
</item 5>

<item 6>
Hoodlum (soundtrack)
Hoodlum is the soundtrack to the 1997 American crime drama film "Hoodlum".

 It was released on August 12, 1997 by Interscope Records and consisted of a blend of hip hop and R&B music.

 The soundtrack peaked at 94 on the "Billboard" 200 and 23 on the Top R&B Albums and contained Mobb Deep's single "Hoodlum" which went to 29 on the Hot Rap
Singles.
</item 6>

<item 7>
Ang Utol Kong Hoodlum
Ang Utol Kong Hoodlum (lit.

 "My Brother is a Hoodlum") is a Filipino drama series developed for TV5 created by Deo J. Fajardo.

 It stars JC de Vera and Jasmine Curtis-Smith.

 It is a remake of the original movie where Robin Padilla played the role of Ben, a hoodlum, and Vina Morales, as Bing.

 The movie was first released in 1991, then a sequel entitled "Miss na Miss Kita: Ang Utol Kong Hoodlum 2" was made the following year.

 It is produced by Vic Del Rosario Jr., and Manuel V. Pangilinan and it also marks as the first primetime series produced by Viva Television for TV5 after a
decade.
</item 7>

<item 8>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 8>

<item 9>
The Yakuza
The Yakuza is a 1974 Japanese-American neo-noir gangster film directed by Sydney Pollack, written by Leonard Schrader, Paul Schrader, and Robert Towne.

 The film is about a man (Robert Mitchum) who returns to Japan after several years away in order to rescue his friend's kidnapped daughter.

 Following a lackluster initial release, the film has since gained a cult following.
</item 9>

<item 10>
Joseph Todaro Jr.
Joseph "Big Joe" Todaro Jr. (born 1945 or 1946) is a Buffalo, New York businessman and former organized crime figure involved in labor racketeering,
loansharking, illegal gambling, narcotics, and murder for hire.

 Joe Todaro Jr. became a business agent for the Laborers' International Union of North America (LIUNA) Local 210.

 In 1984, Joe Todaro Jr. allegedly became the underboss of the Buffalo crime family after his father became the new head of the crime family, upon the
retirement of his predecessor Samuel "Sam the Farmer" Frangiamore.

 In 1990, Joe Todaro Jr. resigned as business agent following investigations on the local's alleged ties to organized crime.

 Outside of organized crime, Todaro operates La Nova Pizzeria, a popular pizza restaurant in Buffalo.
</item 10>
=== END ITEM id=4b05b5781306d08e ===

=== ITEM id=4db4f31e6c7d09de ===
Gold sentences and facts:
Sentence 0: Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and songwriter.
  Fact 0.0: Toshimitsu Deyama was born October 10, 1965 in Tateyama, Chiba.
  Fact 0.1: Toshimitsu Deyama's Japanese name is 出山利三.
  Fact 0.2: Toshimitsu Deyama's name is also written as Deyama Toshimitsu.
  Fact 0.3: Toshimitsu Deyama is known exclusively by his stage name Toshi.
  Fact 0.4: Toshimitsu Deyama is a Japanese singer and songwriter.
Sentence 1:  He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.
  Fact 1.0: Toshimitsu Deyama is best known as the lead vocalist of the heavy metal band X Japan.
  Fact 1.1: Toshimitsu Deyama is a co-founder of X Japan.
  Fact 1.2: X Japan disbanded in 1997.
Sentence 2: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 2.0: Warrel Dane was born March 7, 1969.
  Fact 2.1: Warrel Dane is the lead singer of the American power metal band Sanctuary.

Retrieved items:
<item 1>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 1>

<item 2>
Into the Mirror Live
Into The Mirror Live / Black Reflections is live promotional EP of the band Sanctuary recorded on May 12, 1990 at The Country Club in Reseda, California, during
their "Into the Mirror Black" tour.

 Approximately 1000 CD copies of this were made and less than 500 were ever distributed to radio stations and record stores.

 Because of contractual problems with Epic Records, the full version was never released for public consumption.

 Epic Records destroyed 500+ copies of this CD.

 Warrel Dane has stated on several occasions that he has the master tapes and copyrights of the full show and intended to have it finally released.

 To date that has not happened.

 This CD has been heavily pirated over the years as a 2-on-1 CD with the Satan's Host "Metal From Hell" album on the notorious Reborn Classics record label.
</item 2>

<item 3>
Toshi (musician)
Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and
songwriter.

 He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.

 Toshi then went on to have an extensive solo career.

 In 2007, X Japan reunited and began an ongoing world tour in 2010.

 His distinctive, impassioned tenor vocals have been compared to those of Steve Perry.
</item 3>

<item 4>
The entries cover diverse rock and metal releases, bands, and songs. Highlights include Bad Brains’ acclaimed, genre-blending *I Against I* (1986); Patty
Griffin’s *Impossible Dream* (2004); and Elysia Crampton’s collaborative *Demon City* (2016). Mastodon’s *Leviathan* draws on *Moby-Dick*. Several entries focus
on Nevermore, Sanctuary, and Warrel Dane, including *Dreaming Neon Black*, *
</item 4>

<item 5>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 5>

<item 6>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 6>

<item 7>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 7>

<item 8>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 8>

<item 9>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 9>

<item 10>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 10>
=== END ITEM id=4db4f31e6c7d09de ===

=== ITEM id=9d3ecf7e05d97b69 ===
Gold sentences and facts:
Sentence 0: The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held on March 7, 2015 at the Eko Hotel and Suites, Victoria Island, Lagos, Nigeria.
  Fact 0.0: The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held.
  Fact 0.1: The 2015 AMVCA was held on March 7, 2015.
  Fact 0.2: The 2015 AMVCA was held at the Eko Hotel and Suites.
  Fact 0.3: The Eko Hotel and Suites is in Victoria Island, Lagos, Nigeria.
Sentence 1: Lagos initially emerged as a port city which originated on a collection of islands, which are contained in the present day Local Government Areas (LGAs) of Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa; the islands are separated by creeks, fringing the southwest mouth of Lagos Lagoon, while protected from the Atlantic Ocean by barrier islands and long sand spits such as Bar Beach, which stretch up to 100 km east and west of the mouth.
  Fact 1.0: Lagos initially emerged as a port city.
  Fact 1.1: Lagos originated on a collection of islands.
  Fact 1.2: The islands are contained in the present day Local Government Areas (LGAs) of Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa.
  Fact 1.3: The islands are separated by creeks.
  Fact 1.4: The creeks fringe the southwest mouth of Lagos Lagoon.
  Fact 1.5: The islands are protected from the Atlantic Ocean by barrier islands.
  Fact 1.6: The islands are protected from the Atlantic Ocean by long sand spits such as Bar Beach.
  Fact 1.7: The barrier islands and long sand spits, such as Bar Beach, stretch up to 100 km east and west of the mouth.

Retrieved items:
<item 1>
2013 Africa Magic Viewers Choice Awards
The 2013 Africa Magic Viewers Choice Awards were held on 9 March 2014 and co-hosted by Big Brother Africa presenter IK Osakioduwa and StarGist’s host, Vimbai
Mutinhiri.

 Ivie Okujaiye and Olu Jacobs were honored with the TrailBlazer and Industry Merit Awards respectively.
</item 1>

<item 2>
Africa Magic Viewers' Choice Awards
Africa Magic Viewers' Choice Awards (AMVCA) is an annual accolade presented by Multichoice recognizing outstanding achievement in television and film, voted on
by the general public.

 The inaugural Africa Magic Viewers’ Choice Awards ceremony was held in Lagos, Lagos State in Nigeria on 9 March 2013, and was broadcast live in more than 50
countries.

 Entries into the award ceremony are films and TV series that have been aired on Dstv Channels in the previous year.
</item 2>

<item 3>
The MTV Best New Artist award began in 1984 as Best New Artist in a Video; renamed in 2007, it briefly honored an artist’s yearly work, then returned to
video-based awards and became Artist to Watch (2013–15). Actor-comedian Shaheed Woods is from Jersey City. BBC Somali leadership shifted among non-Somali
managers. Tanzanian actress Elizabeth Michael won major regional awards. The AMVCAs recognize film and TV; launched in 2013, they are
</item 3>

<item 4>
Eric Aghimien
Eric Enomamien Aghimien is a Nigerian director, producer, screenwriter and editor.

 His debut feature film, "A Mile from Home" won awards at both the 2014 Africa Magic Viewers Choice Awards and the 10th Africa Movie Academy Awards.
</item 4>

<item 5>
2015 Africa Magic Viewers Choice Awards
The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held on March 7, 2015 at the Eko Hotel and Suites, Victoria Island, Lagos, Nigeria.

 IK and Vimbai were the hosts of the event.

 A new category, Best Indigenous Language (Igbo), was introduced in the 2015 awards.
</item 5>

<item 6>
2012 British Academy Television Awards
The 2012 British Academy Television Awards (formally known as the Arqiva British Academy Television Awards) were held on 27 May 2012 at the Royal Festival Hall
in London.
</item 6>

<item 7>
British Academy Television Awards 1997
The 1997 British Academy Television Awards were held on 29 April at the Royal Albert Hall in London, as a joint ceremony with the British Academy Film Awards.

 To date, it is the last occasion upon which the two sets awards have been given jointly at the same event; from 1998, the Film and Television Awards were given
at separate ceremonies, for the first time since 1968.
</item 7>

<item 8>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 8>

<item 9>
OC Ukeje
Okechukwu Ukeje, known as OC Ukeje is Nigerian actor, model and musician.

 He came into prominence after winning the Amstel Malta Box Office (AMBO) reality show.

 He has received several awards including Africa Movie Academy Awards, Africa Magic Viewers Choice Awards, Nollywood Movies Awards, Best of Nollywood Awards,
Nigeria Entertainment Awards and Golden Icons Academy Movie Awards.

 He has featured in several award winning films including "Two Brides and a Baby", "Hoodrush", "Alan Poza", "Confusion Na Wa" and "Half of a Yellow Sun".
</item 9>

<item 10>
Africa Magic
Africa Magic is a collection of Pay TV entertainment channels that focus on African programming.

 "Africa Magic", which started off as single channel of the same name, is a brand owned by M-Net and now comprises eight channels.

 The first "Africa Magic" channel was launched in July 2003 as a movie channel and over the next decade, the brand expanded to include seven more channels
comprising movies, television shows and general entertainment.

 Africa Magic currently broadcasts in more than 50 African countries.

 The channels include "Africa Magic Family", "Africa Magic World", "Africa Magic Showcase", "Africa Magic Yoruba", "Africa Magic Igbo", "Africa Magic Hausa".

 "Africa Magic Epic" and "Africa Magic Urban".

 Africa Magic is also responsible for the annual Africa Magic Viewers' Choice Awards (AMVCAs), the biggest celebration of film and television talent in Africa.
</item 10>
=== END ITEM id=9d3ecf7e05d97b69 ===

=== ITEM id=58d54d8c6fe9efc4 ===
Gold sentences and facts:
Sentence 0:  The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.
  Fact 0.0: The album includes a collaboration with Popcaan.
  Fact 0.1: The album includes a collaboration with Zhu.
  Fact 0.2: The album includes a collaboration with Leikeli47.
  Fact 0.3: The album includes a collaboration with Dreezy.
  Fact 0.4: The album includes a collaboration with Flume.
  Fact 0.5: The album includes a collaboration with Pell.
Sentence 1: Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of 2014, signed to Mind of a Genius Records and Columbia Records.
  Fact 1.0: Steven Zhu was born in 1989.
  Fact 1.1: Steven Zhu is known professionally as Zhu.
  Fact 1.2: Zhu is a Chinese American electronic musician.
  Fact 1.3: Zhu is a Chinese American singer.
  Fact 1.4: Zhu has been active since the beginning of 2014.
  Fact 1.5: Zhu is signed to Mind of a Genius Records.
  Fact 1.6: Zhu is signed to Columbia Records.

Retrieved items:
<item 1>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 1>

<item 2>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 2>

<item 3>
Lingdian
Zero Band () are a Chinese rock band from Inner Mongolia.

 The lead singer is Zhou Xiaoou (周晓鸥).

 The English name "Zero Band" was used in Beijing concert advertisements, though the band did not originally use an English name; the name has also been
translated as "Point Zero" in some older English publications.

 The band appeared on the Jingwen label in China, but were licensed outside China by JVC.

 One of the top Beijing bands at the end of the 1990s, Lingdian originally suffered the same resistance from government-owned venues as other indigenous PRC
rock bands.
</item 3>

<item 4>
Zhu (musician)
Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of
2014, signed to Mind of a Genius Records and Columbia Records.

 Until mid-2014, Zhu remained anonymous, asking to be judged by his music alone.

 His debut album, "Generationwhy", was released on 29 July 2016.
</item 4>

<item 5>
Vektroid
Ramona Andra Xavier (born August 19, 1992, known by her primary alias Vektroid) is an American electronic musician from Portland, Oregon.

 She has released music under her primary Vektroid alias, as well as others, such as dstnt, Laserdisc Visions, New Dreams Ltd., Macintosh Plus, Virtual
Information Desk, and PrismCorp Virtual Enterprises.

 Xavier played a prominent role in the popularization of the vaporwave subgenre with the release of her only studio album under her Macintosh Plus alias,
"Floral Shoppe", in 2011.

 The album helped popularize the vaporwave genre throughout the Internet.

 Since then, she has continued to release music through Bandcamp and other online platforms, and has recently collaborated with rapper Siddiq on two albums,
"Midnight Run" and "Palacio Del Rio" (both from 2016).
</item 5>

<item 6>
King Electric
King Electric is Chris Isepp and Peter Hartwig.

 They took their first musical attempts at the end of 1980s / beginning 1990s and first met each other at the upcoming and flourishing Viennese techno scene
back in '92.

 While Hartwig was touring with the well known noise-pop band the Basket Boys, Isepp made first attentions as Electro DJ & Producer.
</item 6>

<item 7>
Kid Confucius
Kid Confucius were an eight-piece Australian band from Sydney, Australia.

 They formed in 2001 and have since played well over 400 live shows around the country, including major festivals as well as their own headline shows at venues
such as The Annandale Hotel and The Metro Theatre.

 In 2005 the band released a self-titled album and two singles "Words" and "Skintight."

 The album, a mish-mash of soul, hip-hop, pop and funk, received great critical acclaim, most notably from Rolling Stone who hailed the album as one of the
standout local releases of the year.

 "Words" also enjoyed some solid months of radio and TV play.

 The band released its second album Stripes in 2007 with three singles "Closer", "Last Straw" and "Moment".

 Stripes was the band's attempt at making a Detroit-era Motown soul album and it was quick to earn rave reviews from press around the country as well as a
publishing deal with Mushroom.

 "Moment" has enjoyed good radio play on triple j. Kid Confucius is set to release its third album in October 2008.

 The first single from this album, "Good Luck", is out now.
</item 7>

<item 8>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 8>

<item 9>
I Remember (AlunaGeorge album)
I Remember is the second studio album by English electronic music duo AlunaGeorge.

 It was released on 16 September 2016 by Island Records.

 The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.

 The album's lead single, "I'm in Control", was released in February 2016, featuring vocals from rapper Popcaan, and peaked at number 39 on the UK Singles
Chart.
</item 9>

<item 10>
Candy Chen
Candy Chen (陳斯亞 born January 3, 1993) is a dancer, actress, host, singer, rapper and a model.

 She is featured on the Taiwanese variety shows: Blackie's Teenage Club and Blackie Lollipop.

 Candy was a member of the Taiwanese girl group Hey Girl from 2010 to 2011, and a member of from 2013 to 2016.

 She is 1/8th Dutch.

 She changed her name to Chen Yi-Ling in July 2012 to follow her mother's maiden name.

 Her English name becomes Nina Chen.

 In November 2013, she changed her name to Candy Chen (English Name) and Chen Si Ya (Chinese Name).
</item 10>
=== END ITEM id=58d54d8c6fe9efc4 ===

=== ITEM id=ea2fbdefbb466526 ===
Gold sentences and facts:
Sentence 0:  He is the main vocalist of the boy group Highlight.
  Fact 0.0: The person described is the main vocalist of the boy group Highlight.
Sentence 1:  She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls' Generation, in 2007.
  Fact 1.0: The person described was a trainee at S.M. Entertainment's Starlight Academy during her middle school years.
  Fact 1.1: The person described debuted as a member of Girls' Generation, the girl group of S.M. Entertainment, in 2007.
  Fact 1.2: The person described debuted in Girls' Generation after her trainee years.

Retrieved items:
<item 1>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 1>

<item 2>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 2>

<item 3>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 3>

<item 4>
Yang Yo-seob discography
The discography of South Korean singer-songwriter, musical actor Yang Yo-seob consists of one extended play and two singles.
</item 4>

<item 5>
Kim Tae-yeon (actress)
Kim Tae-yeon (born January 3, 1976) is a South Korean actress.

 She began her entertainment career as a model, winning Model Line's 40th Fashion Model contest in 1996 and the Pantene Model contest sponsored by Ford Models
in 2000.

 Kim made her film debut in the highly controversial film "Lies" in 1999.
</item 5>

<item 6>
List of songs written by Super Junior
Super Junior (Korean: 슈퍼주니어 ; "Syupeo Junieo") is a South Korean boy band.

 Formed in 2005 by producer Lee Soo-man of S.M. Entertainment, the group comprised a total of thirteen members at its peak.

 Super Junior originally debuted with twelve members, consisting of leader Leeteuk, Heechul, Hangeng, Yesung, Kangin, Shindong, Sungmin, Eunhyuk, Siwon,
Donghae, Ryeowook and Kibum.

 Kyuhyun joined the group in 2006.
</item 6>

<item 7>
Super Junior
Super Junior (Korean: 슈퍼주니어 ; "Syupeo Junieo"), also known as simply SJ or SUJU, is a South Korean boy band.

 Formed in 2005 by producer Lee Soo-man of S.M. Entertainment, the group comprised a total of thirteen members at its peak.

 Super Junior originally debuted with twelve members, consisting of leader Leeteuk, Heechul, Hangeng, Yesung, Kangin, Shindong, Sungmin, Eunhyuk, Siwon,
Donghae, Ryeowook and Kibum.

 Kyuhyun joined the group in 2006.
</item 7>

<item 8>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 8>

<item 9>
Yang Yo-seob
Yang Yo-seob (born January 5, 1990), more commonly known as Yoseob, is a South Korean singer and musical actor.

 He is the main vocalist of the boy group Highlight.
</item 9>

<item 10>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 10>
=== END ITEM id=ea2fbdefbb466526 ===

=== ITEM id=65f84f92ed3b1535 ===
Gold sentences and facts:
Sentence 0:  His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files" (1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
  Fact 0.0: The subject of this sentence portrayed Bret Maverick in the series Maverick from 1957 to 1962.
  Fact 0.1: The subject of this sentence portrayed Bret Maverick in the series Bret Maverick from 1981 to 1982.
  Fact 0.2: The subject of this sentence portrayed Jim Rockford in the series The Rockford Files from 1974 to 1980.
  Fact 0.3: The subject of this sentence portrayed Jim Egan in the series 8 Simple Rules from 2003 to 2005.
Sentence 1: The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.
  Fact 1.0: The Notebook is a 2004 film.
  Fact 1.1: The Notebook is an American romantic drama film.
  Fact 1.2: The Notebook was directed by Nick Cassavetes.
  Fact 1.3: The Notebook is based on a 1996 novel.
  Fact 1.4: The 1996 novel has the same name as the film The Notebook.
  Fact 1.5: The 1996 novel was written by Nicholas Sparks.

Retrieved items:
<item 1>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 1>

<item 2>
Sexual Life
Sexual Life is a 2005 comedy-drama, independent film written and directed by Ken Kwapis, who would go on to chronicle modern romantic life in the better-known
"He's Just Not That Into You" in 2009.

 Produced by Ken Aguado and distributed by Showtime Independent Films.

 Cast members include Azura Skye, Carla Gallo, Anne Heche, Elizabeth Banks, Tom Everett Scott, and Steven Weber.
</item 2>

<item 3>
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 3>

<item 4>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 4>

<item 5>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 5>

<item 6>
Mister Buddwing
Mister Buddwing is a 1966 American film drama directed by Delbert Mann and starring James Garner.
</item 6>

<item 7>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 7>

<item 8>
Milan (2004 film)
Milan is a 2004 romantic drama film released under Star Cinema, ABS-CBN Film Productions, Inc. in the Philippines.

 It stars Claudine Barretto and Piolo Pascual.

 It is a love story written by Raymond Lee, and under Olivia Lamasan's direction.

 The movie made more than P100 million in the box office.
</item 8>

<item 9>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 9>

<item 10>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 10>
=== END ITEM id=65f84f92ed3b1535 ===

=== ITEM id=253552d7064563b3 ===
Gold sentences and facts:
Sentence 0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who successfully worked as the film director, the screenwriter, the art director and the animator.
  Fact 0.0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич) was born on 4 September 1924.
  Fact 0.1: Vladimir Petrovich Danilevich died on 9 October 2001.
  Fact 0.2: Vladimir Petrovich Danilevich was a well-known Soviet animator.
  Fact 0.3: Vladimir Petrovich Danilevich was a well-known Russian animator.
  Fact 0.4: Vladimir Petrovich Danilevich successfully worked as a film director.
  Fact 0.5: Vladimir Petrovich Danilevich successfully worked as a screenwriter.
  Fact 0.6: Vladimir Petrovich Danilevich successfully worked as an art director.
  Fact 0.7: Vladimir Petrovich Danilevich successfully worked as an animator.
Sentence 1: Wolfgang Becker (born 22 June 1954) is a German film director and writer.
  Fact 1.0: Wolfgang Becker was born on 22 June 1954.
  Fact 1.1: Wolfgang Becker is a German film director.
  Fact 1.2: Wolfgang Becker is a German writer.
Sentence 2:  He is best known to the international audience for his work "Good Bye Lenin!
  Fact 2.0: Wolfgang Becker is best known to the international audience for his work "Good Bye Lenin!".
Sentence 3: " (2003).
  Fact 3.0: "Good Bye Lenin!" is from 2003.

Retrieved items:
<item 1>
Farewell (1983 film)
Farewell (Russian: Прощание ) is a 1983 Soviet drama film based on Valentin Rasputin's novel "Farewell to Matyora" and directed by Elem Klimov.
</item 1>

<item 2>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 2>

<item 3>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 3>

<item 4>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 4>

<item 5>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 5>

<item 6>
Good Bye, Lenin!
Good Bye, Lenin!

 is a 2003 German tragicomedy film, directed by Wolfgang Becker.

 The cast includes Daniel Brühl, Katrin Saß, Chulpan Khamatova, and Maria Simon.

 Most scenes were shot at the Karl-Marx-Allee in Berlin and around Plattenbauten near Alexanderplatz.
</item 6>

<item 7>
Wolfgang Becker
Wolfgang Becker (born 22 June 1954) is a German film director and writer.

 He is best known to the international audience for his work "Good Bye Lenin!

" (2003).
</item 7>

<item 8>
Good Bye Lenin! (soundtrack)
Good Bye Lenin!

 is the original soundtrack album of the film of the same title starring Daniel Brühl and Katrin Sass.
</item 8>

<item 9>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 9>

<item 10>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 10>
=== END ITEM id=253552d7064563b3 ===

=== ITEM id=82bcdd405ef27e1e ===
Gold sentences and facts:
Sentence 0: Fastbacks were a Seattle, Washington, punk rock band.
  Fact 0.0: Fastbacks were a punk rock band.
  Fact 0.1: Fastbacks were from Seattle, Washington.
Sentence 1: Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.
  Fact 1.0: Local Natives is an American indie rock band.
  Fact 1.1: Local Natives was previously known as Cavil at Rest.
  Fact 1.2: Local Natives is based in Silver Lake, Los Angeles, California, United States.

Retrieved items:
<item 1>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
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
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 3>

<item 4>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 4>

<item 5>
Glassjaw
Glassjaw is an American post-hardcore band from Long Island, New York.

 The band is fronted by vocalist Daryl Palumbo and guitarist Justin Beck.

 The band has been influential in the progression of the underground music scene in the eastern United States and United Kingdom and on the post-hardcore genre,
and are known for their intense live shows and frequent line-up changes.
</item 5>

<item 6>
Fastbacks
Fastbacks were a Seattle, Washington, punk rock band.

 Formed in 1979 by songwriter/guitarist Kurt Bloch (born August 28, 1960), and friends Lulu Gargiulo (guitar and vocals, born October 12, 1960) and Kim Warnick
(bass and vocals, born April 7, 1959), they disbanded in 2001.

 Their sound mixed a generally punk rock approach to vocals and sound textures with poppy tunes and strong musicianship.
</item 6>

<item 7>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 7>

<item 8>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 8>

<item 9>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 9>

<item 10>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 10>
=== END ITEM id=82bcdd405ef27e1e ===

=== ITEM id=83c325802bbe5f64 ===
Gold sentences and facts:
Sentence 0: Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban parents in France, where she was also raised.
  Fact 0.0: Anaïs Nin was born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell.
  Fact 0.1: Anaïs Nin was born on February 21, 1903.
  Fact 0.2: Anaïs Nin died on January 14, 1977.
  Fact 0.3: Anaïs Nin was an essayist.
  Fact 0.4: Anaïs Nin was a memoirist.
  Fact 0.5: Anaïs Nin was born to Cuban parents.
  Fact 0.6: Anaïs Nin was born in France.
  Fact 0.7: Anaïs Nin was raised in France.
Sentence 1: Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as a major exponent of romantic modernism in Greece and the world.
  Fact 1.0: Odysseas Elytis is a pen name of Odysseas Alepoudellis.
  Fact 1.1: Odysseas Elytis's Greek name is Οδυσσέας Ελύτης.
  Fact 1.2: Odysseas Alepoudellis's Greek name is Οδυσσέας Αλεπουδέλλης.
  Fact 1.3: Odysseas Elytis was born on 2 November 1911.
  Fact 1.4: Odysseas Elytis died on 18 March 1996.
  Fact 1.5: Odysseas Elytis was regarded as a major exponent of romantic modernism in Greece.
  Fact 1.6: Odysseas Elytis was regarded as a major exponent of romantic modernism in the world.
Sentence 2:  In 1979 he was awarded the Nobel Prize in Literature.
  Fact 2.0: Odysseas Elytis was awarded the Nobel Prize in Literature in 1979.

Retrieved items:
<item 1>
The entries survey literature, theatre, film, music, and curiosities: French Decadence favored artifice and excess; Camus staged Dostoyevsky’s *The Possessed*
in 1959; Algren won the 1950 National Book Award; Poe satirized Gothic fiction; Moorcock wrote Elric and science-fiction novels; Nin’s diaries and erotica
appeared partly posthumously. Wilde’s *Dorian Gray* faced censorship and condemnation. Verne’s
</item 1>

<item 2>
Anaïs Nin (1903–1977), a French-born writer of Cuban parentage, produced journals, novels, essays, criticism, stories, and erotica; much of her work appeared
posthumously. Her published diary began in 1914 as a letter to her absent father. Unexpurgated diary volumes include *Henry and June* (1931–32), *Incest*
(1932–34), *Fire* (1934–37), and *Nearer
</item 2>

<item 3>
Seduction of the Minotaur
Seduction of the Minotaur is an autobiographical novel by the mixed nationality writer Anaïs Nin, the last part of her "Cities of the Interior" sequence.

 It is about a woman named Lillian, and her self-psychoanalysis.

 The setting is taken from Anaïs' diary account of her first trip to Acapulco in 1947, and the novel repeats much of the first part of "The Diary of Anaïs Nin"
volume V.

 Since the author was concerned with psychology rather than physical adventure, there is actually less violence in the novel than in the diary account.

 The exception is that the doctor allows himself to be shot because he is loved only as a doctor and never as a man, perhaps patterned after her understanding
of Otto Rank's death.
</item 3>

<item 4>
Incest: From a Journal of Love
Incest: From a Journal of Love: The Unexpurgated Diary of Anaïs Nin (1932–1934) is a 1992 non-fiction book by Anaïs Nin.

 It is a continuation of the diary entries first published in "Henry and June: From the Unexpurgated Diary of Anaïs Nin".

 It features Nin's relationships with writer Henry Miller, his wife June Miller, the psychoanalyst Otto Rank, her father Joaquín Nin, and her husband Hugh
Parker Guiler.

 She also copied some of her correspondence with these people into her diary.

 Much of this book was written in English, although those of her letters which were originally written in French and Spanish were translated.

 Most of this diary takes place in France, particularly Clichy, Paris and Louveciennes.
</item 4>

<item 5>
Anaïs Nin
Anaïs Nin (] ; born Angela Anaïs Juana Antolina Rosa Edelmira Nin y Culmell; February 21, 1903 – January 14, 1977) was an essayist and memoirist born to Cuban
parents in France, where she was also raised.

 She spent some time in Spain and Cuba, but lived most of her life in the United States, where she became an established author.

 She wrote journals (which span more than 60 years, beginning when she was 11 years old and ending shortly before her death), novels, critical studies, essays,
short stories, and erotica.

 A great deal of her work, including "Delta of Venus" and "Little Birds", was published posthumously.
</item 5>

<item 6>
Nearer the Moon
Nearer the Moon: From a Journal of Love (full title "Nearer the Moon: From a Journal of Love, the Unexpurgated Diary of Anaïs Nin (1937–1939)") is a 1996 book
based upon material excerpted from the unpublished diaries of Anaïs Nin.

 It corresponds temporally to part of Nin's published diaries.

 It consists mainly of material that was left out of the published version because it would have hurt people involved or their relationships with Anaïs Nin had
it been published at the time.
</item 6>

<item 7>
The Diary of Anaïs Nin
The Diary of Anaïs Nin is the published version of Anaïs Nin's own private manuscript diary, which she began at age 11 in 1914 during a trip from Europe to New
York with her mother and two brothers.

 Anaïs Nin would later say she had begun the diary as a letter to her father, Cuban composer Joaquín Nin, who had abandoned the family a few years earlier.
</item 7>

<item 8>
Collages (novel)
Collages, published in 1964, was Anaïs Nin's last published novel (excluding her erotica).

 It is very different from the previous novels of the "Cities of the Interior" series, because it contains none of the familiar characters in those novels.

 It is also different in that it takes place on two continents, has about two dozen important characters, and for the first time in Anaïs Nin's work the female
lead is not seeking psychological wholeness.

 Most of Anaïs Nin's other novels only have a few main characters, there is rarely much geographical movement in the other novels, and usually the lead
character is a woman seeking psychological wholeness.
</item 8>

<item 9>
Anaïs Nin: A Biography
Anaïs Nin: A Biography is Deirdre Bair's award-winning biography of writer Anaïs Nin.

 It is considered arguably by many to be the most comprehensive, well-researched, and scholarly biography available of Nin.

 Though the biography has received praise, it has also angered some fans of Anaïs Nin as well as some of her former associates, some of whom claim that Bair's
critical and rigorous investigation of Anaïs Nin's life is unsympathetic.
</item 9>

<item 10>
Odysseas Elytis
Odysseas Elytis (Greek: Οδυσσέας Ελύτης , ] , pen name of Odysseas Alepoudellis, Greek: Οδυσσέας Αλεπουδέλλης ; 2 November 1911 – 18 March 1996) was regarded as
a major exponent of romantic modernism in Greece and the world.

 In 1979 he was awarded the Nobel Prize in Literature.
</item 10>
=== END ITEM id=83c325802bbe5f64 ===

=== ITEM id=96a9d5e762dee1dd ===
Gold sentences and facts:
Sentence 0: Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).
  Fact 0.0: Blinding Edge Pictures is an American film production company.
  Fact 0.1: Blinding Edge Pictures was founded in 2000 by M. Night Shyamalan.
  Fact 0.2: Blinding Edge Pictures is known for producing films written and directed by Shyamalan.
  Fact 0.3: Blinding Edge Pictures produced Unbreakable (2000), written and directed by Shyamalan.
  Fact 0.4: Blinding Edge Pictures produced Signs (2002), written and directed by Shyamalan.
  Fact 0.5: Blinding Edge Pictures produced The Village (2004), written and directed by Shyamalan.
  Fact 0.6: Blinding Edge Pictures produced The Happening (2008), written and directed by Shyamalan.
  Fact 0.7: Blinding Edge Pictures produced After Earth (2013), written and directed by Shyamalan.
  Fact 0.8: Blinding Edge Pictures produced The Visit (2015), written and directed by Shyamalan.
  Fact 0.9: Blinding Edge Pictures produced Split (2017), written and directed by Shyamalan.
Sentence 1: Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson, alongside Robin Wright and Spencer Treat Clark.
  Fact 1.0: Unbreakable is a 2000 American superhero thriller film.
  Fact 1.1: Unbreakable was written by M. Night Shyamalan.
  Fact 1.2: Unbreakable was produced by M. Night Shyamalan.
  Fact 1.3: Unbreakable was directed by M. Night Shyamalan.
  Fact 1.4: Unbreakable stars Bruce Willis.
  Fact 1.5: Unbreakable stars Samuel L. Jackson.
  Fact 1.6: Robin Wright is in the cast of Unbreakable.
  Fact 1.7: Spencer Treat Clark is in the cast of Unbreakable.

Retrieved items:
<item 1>
Mercury Rising
Mercury Rising is a 1998 American political action thriller film starring Bruce Willis and Alec Baldwin.

 Directed by Harold Becker, the movie is based on Ryne Douglas Pearson's 1996 novel originally published as "Simple Simon".

 Willis plays Art Jeffries, an undercover FBI agent who protects a 9-year-old boy with autism who is targeted by government assassins after he cracks a top
secret government code.
</item 1>

<item 2>
Cop Out (2010 film)
Cop Out is a 2010 American buddy cop action-comedy film directed and edited by Kevin Smith, written by Mark and Robb Cullen and starring Bruce Willis, Tracy
Morgan, Kevin Pollak and Seann William Scott.

 The plot revolves around two veteran NYPD partners (Willis and Morgan) on the trail of a stolen, rare, mint-condition baseball card who find themselves up
against a relentless, memorabilia-obsessed bloodthirsty gangster.

 This is the first film that Smith directed that he did not also write.

 Upon its release, the film was met with negative reviews by critics and underperformed at the box office.

 It is nevertheless Kevin Smith's highest grossing film.
</item 2>

<item 3>
Blinding Edge Pictures
Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by
Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).

 In 2015, the company released its first television series "Wayward Pines".
</item 3>

<item 4>
Signs (film)
Signs is a 2002 American science fiction horror film written and directed by M. Night Shyamalan and executive produced by Shyamalan, Frank Marshall, Kathleen
Kennedy and Sam Mercer.

 A joint collective effort to commit to the film's production was made by Blinding Edge Pictures and The Kennedy/Marshall Company.

 It was commercially distributed by Touchstone Pictures theatrically, and by Touchstone Home Entertainment in home media format.

 Its story focuses on a former Episcopal priest named Graham Hess, played by Mel Gibson, who discovers a series of crop circles in his cornfield.

 Hess slowly discovers that the phenomena are a result of extraterrestrial life.

 It also stars Joaquin Phoenix, Rory Culkin, and Abigail Breslin.

 "Signs" explores faith, kinship, and extraterrestrials.
</item 4>

<item 5>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 5>

<item 6>
Striking Distance
Striking Distance is a 1993 American action thriller film starring Bruce Willis as Pittsburgh Police homicide detective Thomas Hardy.

 The film co-stars Sarah Jessica Parker, Dennis Farina, and Tom Sizemore.

 It was directed by Rowdy Herrington and written by Herrington and Marty Kaplan.

 The film was shot on location throughout Pittsburgh; its early title was "Three Rivers".
</item 6>

<item 7>
Unbreakable (film)
Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson,
alongside Robin Wright and Spencer Treat Clark.

 The movie is the first installment in a trilogy.

 In "Unbreakable", a security guard named David Dunn survives a horrific train crash.

 After the incident, with the help of a manipulative disabled comic book shop owner named Elijah Price, he learns that he possesses superhuman powers.

 As Dunn explores and reluctantly confronts his powers while trying to navigate a difficult family life, he begins to fight crime and learns the true nature of
Elijah Price.
</item 7>

<item 8>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 8>

<item 9>
Last Man Standing (1996 film)
Last Man Standing is a 1996 American action thriller film written and directed by Walter Hill and starring Bruce Willis, Christopher Walken and Bruce Dern.

 It is a credited remake of Akira Kurosawa's "Yojimbo".
</item 9>

<item 10>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 10>
=== END ITEM id=96a9d5e762dee1dd ===

=== ITEM id=40e7700627ba1146 ===
Gold sentences and facts:
Sentence 0: Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All Stars, as well of his own solo band, the Kuwata band.
  Fact 0.0: Keisuke Kuwata's name is written 桑田 佳祐 (Kuwata Keisuke).
  Fact 0.1: Keisuke Kuwata was born February 26, 1956.
  Fact 0.2: Keisuke Kuwata is a Japanese multi-instrumentalist.
  Fact 0.3: Keisuke Kuwata is a Japanese singer-songwriter.
  Fact 0.4: Keisuke Kuwata is frontman for the Southern All Stars.
  Fact 0.5: Keisuke Kuwata has his own solo band, the Kuwata band.
Sentence 1: Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.
  Fact 1.0: Moya Brennan was born Máire Ní Bhraonáin.
  Fact 1.1: Moya Brennan is also known as Máire Brennan.
  Fact 1.2: Moya Brennan was born 4 August 1952.
  Fact 1.3: Moya Brennan is an Irish folk singer.
  Fact 1.4: Moya Brennan is an Irish songwriter.
  Fact 1.5: Moya Brennan is an Irish harpist.
  Fact 1.6: Moya Brennan is an Irish philanthropist.

Retrieved items:
<item 1>
Kevin Mayer
Kevin Mayer (] , ] or ] , born 10 February 1992) is a French athlete specializing in combined events.

 A native of Drôme in southeast France, Mayer excelled in combined events from a young age; highlights from his youth career include winning gold medals in the
octathlon at the 2009 World Youth Championships and in the decathlon at the 2010 World Junior Championships and at the 2011 European Athletics Junior
Championships, establishing a new national junior record at the latter.
</item 1>

<item 2>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 2>

<item 3>
Two Horizons
Two Horizons is the fifth solo album by Irish singer Moya Brennan.

 It is her first full-length release under the name "Moya Brennan", as opposed to "Máire Brennan".

 The album was predominantly recorded in her home studio in Dublin and was nominated for a Grammy award.

 The album was recorded between 2002–2003 and first became available on 23 October 2003.

 It is also Brennan's first non-Christian album since her 1993 recording Misty Eyed Adventures.

 It is the most successful of her solo albums to date and the second to be Grammy-nominated.
</item 3>

<item 4>
Kate O'Beirne
Kate Walsh O'Beirne (September 23, 1949 – April 23, 2017) was the President of National Review Institute.

 She was the Washington editor of "National Review".

 Her column, "Bread and Circuses," covered Congress, politics, and U.S. domestic policy.
</item 4>

<item 5>
Koi Kaze
Koi Kaze (Japanese: 恋風 , lit.

 "Love Wind") is a Japanese manga series written and illustrated by Motoi Yoshida.

 It premiered in the September 2001 issue of "Evening" and ran for 29 chapters until its conclusion in the October 2004 issue.

 The individual chapters were collected and published in five "tankōbon" volumes by Kodansha.

 The series tells of the love affair that develops between 27-year-old Koshiro, and his 15-year-old sister, Nanoka.
</item 5>

<item 6>
Moya Brennan
Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.

 She is the older sister of Enya and Brídín Brennan.

 She began performing professionally in 1970 when her family formed the band Clannad, and is considered as the "First Lady of Celtic Music".

 Moya released her first solo album in 1992 called "Máire", a successful venture.

 She has been nominated for two Grammys and has won an Emmy Award.

 She has recorded music for several soundtracks, including "Titanic", "To End All Wars" and "King Arthur".
</item 6>

<item 7>
Bruce Yonemoto
Bruce Yonemoto (born 1949) is a Japanese-American multimedia artist.

 His work has been recognized by the National Endowment for the Arts, the American Film Institute, The Rockefeller Foundation, the Maya Deren Award for
Experimental Film and Video, and a mid-career survey show at the Japanese American National Museum, in addition to major solo exhibitions at the
InterCommunication Center in Tokyo, the Institute of Contemporary Art the University of Pennsylvania, and the Kemper Museum in Kansas City, MO.

 Yonemoto was featured in the 2002 Corcoran Gallery Bienniel, in Washington D.C. Yonemoto was also a recipient of the Creative Capital Visual Arts Award in
2008.
</item 7>

<item 8>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 8>

<item 9>
2014 Monterrey Open – Doubles
Tímea Babos and Kimiko Date-Krumm were the defending champions, but chose not to participate together.

 Babos played alongside Olga Govortsova, while Date-Krumm teamed up with Karolína Plíšková.

 The two teams were scheduled to meet in the semifinals, but Date-Krumm withdrew with a right leg injury.

<br>
</item 9>

<item 10>
My Match Is a Makin'
My Match Is A Makin' (English translation of opening track title) is a music album by Irish musicians Moya Brennan and Cormac de Barra.

 This is Moya's eighth studio album to be released.

 It was released on 14 April 2010 exclusively to concert goers on her Spring 2010 tour of the Netherlands.

 The album was re-released to a wider audience under the title Voices & Harps.
</item 10>
=== END ITEM id=40e7700627ba1146 ===

=== ITEM id=582eaeb745cf55c7 ===
Gold sentences and facts:
Sentence 0: Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.
  Fact 0.0: Alice Malsenior Walker was born on February 9, 1944.
  Fact 0.1: Alice Malsenior Walker is an American novelist.
  Fact 0.2: Alice Malsenior Walker is an American short story writer.
  Fact 0.3: Alice Malsenior Walker is an American poet.
  Fact 0.4: Alice Malsenior Walker is an American activist.
Sentence 1: Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.
  Fact 1.0: Michael David Herr was born on April 13, 1940.
  Fact 1.1: Michael David Herr died on June 23, 2016.
  Fact 1.2: Michael David Herr was an American writer.
  Fact 1.3: Michael David Herr was an American war correspondent.
  Fact 1.4: Michael David Herr is known as the author of the memoir 'Dispatches', published in 1977.
  Fact 1.5: 'Dispatches' is a memoir of Michael David Herr's time as a correspondent for 'Esquire magazine' from 1967 to 1969.
  Fact 1.6: Michael David Herr was a correspondent during the Vietnam War.

Retrieved items:
<item 1>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 1>

<item 2>
Tobe Levin
Alice Walker expressed appreciation for the text that shows worldwide solidarity with the novelist's literary abolition efforts in the early nineties.

 Levin has also teamed up with Maria Kiminta and photographer Britta Radike to publish a memoir and sourcebook, "Kiminta.

 A Maasai's Fight against Female Genital Mutilation."
</item 2>

<item 3>
Alice Walker: Beauty in Truth
Alice Walker: Beauty in Truth is a documentary film directed by Pratibha Parmar, made by Kali Films production company.

 The film follows the life of the Pulitzer Prize-winning author, poet and activist Alice Walker.

 Shooting began in May 2011.

 It was aired on the BBC on Monday July 8, 2013, and on PBS on February 7, 2014.
</item 3>

<item 4>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 4>

<item 5>
Haniel Long
Haniel Clark Long (March 9, 1888 – October 17, 1956) was an American poet, novelist, publisher and academic.

 He is best known for his novella, "Interlinear to Cabeza de Vaca" (1936), a fictionalized account of the true story of a Spanish conquistador in 16th century
North America.
</item 5>

<item 6>
Apocalypse Now
Apocalypse Now is a 1979 American epic war film directed, produced, and co-written by Francis Ford Coppola.

 It was co-written by John Milius with narration written by Michael Herr.

 It stars Marlon Brando, Robert Duvall, Martin Sheen, Frederic Forrest, Albert Hall, Sam Bottoms, Larry Fishburne, and Dennis Hopper.

 The screenplay, written by Milius, adapts the story of Joseph Conrad's novella "Heart of Darkness", changing its setting from late nineteenth-century Congo to
the Vietnam War.

 It draws from Herr's "Dispatches" and Werner Herzog's "Aguirre, the Wrath of God" (1972).

 The film revolves around Captain Benjamin L. Willard (Sheen) on a secret mission to assassinate Colonel Kurtz, a renegade who is presumed insane.
</item 6>

<item 7>
Tobe Levin
Tobe Levin Freifrau von Gleichen (*Feb 16, 1948), a multi-lingual scholar, translator, editor and activist, is an Associate of the Hutchins Center for African
and African American Research at Harvard University; a Visiting Research Fellow at the International Gender Studies Centre, Lady Margaret Hall, University of
Oxford; an activist against female genital mutilation (FGM) and professor of English Emerita at the University of Maryland, University College.

 Having received her PhD in 1979 from Cornell University, she is most known for combining her advocacy against FGM with her academic scholarship in comparative
literature.

 She has published peer-reviewed and popular articles and book chapters, edited four books, launched UnCUT/VOICES Press in 2009 and founded "Feminist Europa
Review of Books" (1998-2010).

 Her most notable works to date are "Empathy and Rage.

 Female Genital Mutilation in African Literature" and " Waging Empathy.

 Alice Walker, Possessing the Secret of Joy, and the Global Movement to Ban FGM."
</item 7>

<item 8>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 8>

<item 9>
Dispatches (book)
Dispatches is a New Journalism book by Michael Herr that describes the author's experiences in Vietnam as a war correspondent for "Esquire" magazine.

 First published in 1977, "Dispatches" was one of the first pieces of American literature that portrayed the experiences of soldiers in the Vietnam War for
American readers.
</item 9>

<item 10>
Warrior Marks
Warrior Marks: Female Genital Mutilation and the Sexual Blinding of Women is a 1993 book by Alice Walker with Pratibha Parmar, who made an award-winning
documentary of the same name.

 Following on from her 1992 novel "Possessing the Secret of Joy", Walker undertakes a journey to parts of Africa where clitoridectomy is still practised.

 "Warrior Marks" is a harrowing work as Walker interviews women who have had the operation done and finally interviews a woman—circumcised herself—who performs
the operation.
</item 10>
=== END ITEM id=582eaeb745cf55c7 ===
