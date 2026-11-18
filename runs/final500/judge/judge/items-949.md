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

This file has 7 items.

=== ITEM id=7d7ed74bda2ca27d ===
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
Domain name
Domain name - wikipedia Domain name Jump to : navigation , search This article is about domain names in the Internet . For other uses , see Domain . The
hierarchy of labels in a fully qualified domain name . A domain name is an identification string that defines a realm of administrative autonomy , authority or
control within the Internet . Domain names are formed by the rules and procedures of the Domain Name System ( DNS ) . Any name registered in the DNS is a domain
name . Domain names are used in various networking contexts and for application - specific naming and addressing purposes . In general , a domain name
represents an Internet Protocol ( IP ) resource , such as a personal computer used to access the Internet , a server computer hosting a web site , or the web
site itself or any other service communicated via the Internet . In 2017 , 330.6 million domain names had been registered . Domain names are organized in
subordinate levels ( subdomains ) of the DNS root domain , which is nameless .
</item 1>

<item 2>
Domain name
For special purposes , such as network testing , documentation , and other applications , IANA also reserves a set of special - use domain names . This list
contains domain names such as example , local , localhost , and test . Other top - level domain names containing trade marks are registered for corporate use .
Cases include brands such as BMW , Google , and Canon . Second - level and lower level domains Below the top - level domains in the domain name hierarchy are
the second - level domain ( SLD ) names . These are the names directly to the left of . com , . net , and the other top - level domains . As an example , in the
domain example.co.uk , co is the second - level domain . Next are third - level domains , which are written immediately to the left of a second - level domain .
There can be fourth - and fifth - level domains , and so on , with virtually no limitation . An example of an operational domain name with four levels of domain
labels is sos.state.oh.us . Each label is separated by a full stop ( dot ) . '
</item 2>

<item 3>
Domain name
Hostnames impose restrictions on the characters allowed in the corresponding domain name . A valid hostname is also a valid domain name , but a valid domain
name may not necessarily be valid as a hostname . Top - level domains The top - level domains ( TLDs ) such as com , net and org are the highest level of domain
names of the Internet . Top - level domains form the DNS root zone of the hierarchical Domain Name System . Every domain name ends with a top - level domain
label . When the Domain Name System was devised in the 1980s , the domain name space was divided into two main groups of domains . The country code top - level
domains ( ccTLD ) were primarily based on the two - character territory codes of ISO - 3166 country abbreviations . In addition , a group of seven generic top -
level domains ( gTLD ) was implemented which represented a set of categories of names and multi-organizations . These were the domains gov , edu , com , mil ,
org , net , and int . During the growth of the Internet , it became desirable to create additional generic top - level domains .
</item 3>

<item 4>
Domain name
The first - level set of domain names are the top - level domains ( TLDs ) , including the generic top - level domains ( gTLDs ) , such as the prominent domains
com , info , net , edu , and org , and the country code top - level domains ( ccTLDs ) . Below these top - level domains in the DNS hierarchy are the second -
level and third - level domain names that are typically open for reservation by end - users who wish to connect local area networks to the Internet , create
other publicly accessible Internet resources or run web sites . The registration of these domain names is usually administered by domain name registrars who
sell their services to the public . A fully qualified domain name ( FQDN ) is a domain name that is completely specified with all labels in the hierarchy of the
DNS , having no parts omitted . Labels in the Domain Name System are case - insensitive , and may therefore be written in any desired capitalization method ,
but most commonly domain names are written in lowercase in technical contexts .
</item 4>

<item 5>
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
</item 5>

<item 6>
Domain name
Use in web site hosting The domain name is a component of a uniform resource locator ( URL ) used to access web sites , for example : URL :
http://www.example.net/index.html Top - level domain : net Second - level domain : example Host name : www A domain name may point to multiple IP addresses to
provide server redundancy for the services offered , a feature that is used to manage the traffic of large , popular web sites . Web hosting services , on the
other hand , run servers that are typically assigned only one or a few addresses while serving websites for many domains , a technique referred to as virtual
web hosting . Such IP address overloading requires that each request identifies the domain name being referenced , for instance by using the HTTP request header
field Host : , or Server Name Indication . Abuse and regulation Critics often claim abuse of administrative power over domain names . Particularly noteworthy
was the VeriSign Site Finder system which redirected all unregistered . com and . net domains to a VeriSign webpage .
</item 6>

<item 7>
Domain name
Domain name space Today , the Internet Corporation for Assigned Names and Numbers ( ICANN ) manages the top - level development and architecture of the Internet
domain name space . It authorizes domain name registrars , through which domain names may be registered and reassigned . The hierarchical domain name system ,
organized into zones , each served by domain name servers . The domain name space consists of a tree of domain names . Each node in the tree holds information
associated with the domain name . The tree sub-divides into zones beginning at the DNS root zone . Domain name syntax A domain name consists of one or more
parts , technically called labels , that are conventionally concatenated , and delimited by dots , such as example.com . The right-most label conveys the top -
level domain ; for example , the domain name www.example.com belongs to the top - level domain com . The hierarchy of domains descends from the right to the
left label in the name ; each label to the left specifies a subdivision , or subdomain of the domain to the right .
</item 7>

<item 8>
Domain name
For example : the label example specifies a node example.com as a subdomain of the com domain , and www is a label to create www.example.com , a subdomain of
example.com . Each label may contain from 1 to 63 octets . The empty label is reserved for the root node and when fully qualified is expressed as the empty
label terminated by a dot . The full domain name may not exceed a total length of 253 ASCII characters in its textual representation . Thus , when using a
single character per label , the limit is 127 levels : 127 characters plus 126 dots have a total length of 253 . In practice , some domain registries may have
shorter limits . A hostname is a domain name that has at least one associated IP address . For example , the domain names www.example.com and example.com are
also hostnames , whereas the com domain is not . However , other top - level domains , particularly country code top - level domains , may indeed have an IP
address , and if so , they are also hostnames . Hostnames impose restrictions on the characters allowed in the corresponding domain name .
</item 8>

<item 9>
Domain name
Each label is separated by a full stop ( dot ) . ' sos ' is said to be a sub-domain of ' state.oh.us ' , and ' state ' a sub-domain of ' oh.us ' , etc . In
general , subdomains are domains subordinate to their parent domain . An example of very deep levels of subdomain ordering are the IPv6 reverse resolution DNS
zones , e.g. , 1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa , which is the reverse DNS resolution domain name for the IP address of
a loopback interface , or the localhost name . Second - level ( or lower - level , depending on the established parent hierarchy ) domain names are often
created based on the name of a company ( e.g. , bbc.co.uk ) , product or service ( e.g. hotmail.com ) . Below these levels , the next domain name component has
been used to designate a particular host server .
</item 9>

<item 10>
Domain name
Labels are always separated by the full stop ( period ) character in the textual name representation . Business models Domain names are often seen in analogy to
real estate in that domain names are foundations on which a website can be built , and the highest quality domain names , like sought - after real estate , tend
to carry significant value , usually due to their online brand - building potential , use in advertising , search engine optimization , and many other criteria
. A few companies have offered low - cost , below - cost or even free domain registration with a variety of models adopted to recoup the costs to the provider .
These usually require that domains be hosted on their website within a framework or portal that includes advertising wrapped around the domain holder 's content
, revenue from which allows the provider to recoup the costs . Domain registrations were free of charge when the DNS was new . A domain holder may provide
infinite number of subdomains in their domain . For example , the owner of example.org could provide subdomains such as foo.example.org and foo.bar.example.org
to interested parties . Many desirable domain names are already assigned and users must search for other acceptable names , using Web - based search features ,
or WHOIS and dig operating system tools .
</item 10>
=== END ITEM id=7d7ed74bda2ca27d ===

=== ITEM id=78f9daa6a8364491 ===
Gold sentences and facts:
Sentence 0: In August 2015 , the series was reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop other series with Marvel instead .
  Fact 0.0: In August 2015, the series was reinvented as Most Wanted.
  Fact 0.1: The series Most Wanted received a pilot order from ABC in August 2015.
  Fact 0.2: The cast was filled out in early 2016.
  Fact 0.3: Production was completed by late March.
  Fact 0.4: The episode was co-produced by ABC Studios and Marvel Television.
  Fact 0.5: In May 2016, ABC announced that it would not pick up the pilot to series.
  Fact 0.6: In May 2016, ABC announced that it would look to develop other series with Marvel instead.

Retrieved items:
<item 1>
Most Wanted (TV pilot)
Most Wanted ( TV pilot ) - wikipedia Most Wanted ( TV pilot ) Jump to : navigation , search Most Wanted Genre Action / Adventure Science fiction Superhero Drama
Created by Jeffrey Bell Paul Zbyszewski Based on Mockingbird by Gerry Conway Barry Smith Lance Hunter by Gary Friedrich Herb Trimpe Starring Adrianne Palicki
Nick Blood Delroy Lindo Fernanda Andrade Oded Fehr Country of origin United States Original language ( s ) English Production Executive producer ( s ) Jeffrey
Bell Paul Zbyszewski Jeph Loeb Jim Chory Production company ( s ) ABC Studios Marvel Television Distributor Disney -- ABC Domestic Television Release Original
network ABC Picture format 720p ( HDTV ) Audio format 5.1 surround sound Chronology Related shows Marvel 's Agents of S.H.I.E.L.D. Marvel Cinematic Universe
television series Marvel 's Most Wanted , or simply Most Wanted , is an unaired American television pilot based on the Marvel Comics characters Bobbi Morse and
Lance Hunter .
</item 1>

<item 2>
Most Wanted (TV pilot)
Retrieved March 25 , 2016 . ^ Jump up to : Abrams , Natalie ( July 25 , 2014 ) . `` Agents of S.H.I.E.L.D. Reveals New Characters , Carter Adds Captain
Directors '' . TV Guide . Archived from the original on July 26 , 2014 . Retrieved July 25 , 2014 . ^ Jump up to : Andreeva , Nellie ( January 26 , 2016 ) . ``
' Marvel 's Most Wanted ' ABC Pilot Casts Delroy Lindo As Dominic Fortune '' . Deadline.com . Archived from the original on January 26 , 2016 . Retrieved
January 27 , 2016 . Jump up ^ Huver , Scott ( April 22 , 2016 ) . `` '' Marvel 's Most Wanted '' Co-Star Andrade Promises An `` Unpredictable '' Series `` .
Comic Book Resources . Archived from the original on April 22 , 2016 . Retrieved April 22 , 2016 . ^ Jump up to : `` Marvel 's Most Wanted '' .
</item 2>

<item 3>
Most Wanted (TV pilot)
Retrieved August 9 , 2016 . External links ( edit ) Marvel 's Most Wanted on IMDb Marvel Cinematic Universe Feature films Phase One Iron Man The Incredible Hulk
Iron Man 2 Thor Captain America : The First Avenger The Avengers accolades Phase Two Iron Man 3 Thor : The Dark World Captain America : The Winter Soldier
Guardians of the Galaxy Avengers : Age of Ultron Ant - Man Phase Three Captain America : Civil War Doctor Strange Guardians of the Galaxy Vol. 2 Spider - Man :
Homecoming Thor : Ragnarok Black Panther Avengers : Infinity War production Ant - Man and the Wasp Untitled Avengers film production Television series ABC
Agents of S.H.I.E.L.D. episodes season 1 5 characters Agent Carter season 1 characters Inhumans `` Behold ... The Inhumans '' `` Those Who Would Destroy Us ''
Netflix Daredevil season 1 characters Jessica Jones season 1 characters Luke Cage season 1 characters Iron Fist season 1 characters The Defenders The Punisher
Hulu Runaways Freeform Cloak & Dagger Unaired pilot Most Wanted Other Short films Item 47 Agent Carter All Hail the King Digital series WHIH Newsfront Agents of
S.H.I.E.L.D.
</item 3>

<item 4>
Most Wanted (TV pilot)
`` ' Marvel 's Most Wanted ' Not Going Forward at ABC '' . Variety . Archived from the original on May 13 , 2016 . Retrieved May 13 , 2016 . Jump up ^ Topel ,
Fred ( August 4 , 2016 ) . `` Why ABC Passed on ' Marvel 's Most Wanted ' and ' Agent Carter ' Season 3 ( TCA 2016 ) '' . / Film . Archived from the original on
August 5 , 2016 . Retrieved August 6 , 2016 . Jump up ^ Topel , Fred ( August 8 , 2016 ) . `` Marvel TV 's Jeph Loeb on Fox 's `` Reimagined '' ' Hellfire Club
' Series , ' Cloak and Dagger ' , and More ( TCA 2016 ) `` . / Film . Archived from the original on August 9 , 2016 . Retrieved August 9 , 2016 .
</item 4>

<item 5>
Most Wanted (TV pilot)
In August 2015 , the spin - off series received new life , reworked as Marvel 's Most Wanted and given a pilot order . Bell and Zbyszewski were again developing
the series , writing the pilot and executive producing . The pair were set to serve as showrunners for the series , if it was picked up . Loeb and Jim Chory
were also attached as executive producers . The series would still focus on Morse and Hunter , with Palicki and Blood both attached , and was described as `` a
new take focusing on the same duo and their continuing adventures . '' In January 2016 , Lee stated that the pilot was `` absolutely '' moving forward ,
praising the script and saying production would begin `` in the next few months . '' In the Agents of S.H.I.E.L.D. episode `` Parting Shot '' , the two
characters were written off that series , having them leave and disavow S.H.I.E.L.D.
</item 5>

<item 6>
Most Wanted (TV pilot)
^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) . `` How Will ' Agents of SHIELD ' Set Up Spinoff ' Marvel 's Most Wanted ' ? '' . The Hollywood Reporter
. Archived from the original on August 20 , 2016 . Retrieved August 20 , 2016 . Jump up ^ Carinacas , Peter ( February 17 , 2016 ) . `` Artisans Booked on '
Marvel 's Most Wanted , ' ' Pearl , ' ' Dirty Dancing ' and ' Stronger ' '' . Variety . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017
. Jump up ^ Carinacas , Peter ( March 15 , 2016 ) . `` Agencies Book Artisans on ' Thor : Ragnarok , ' ' Sugar Queen , ' ' Chagrin Falls ' and ' Bunker Hill '
'' . Variety . Archived from the original on March 20 , 2016 . Retrieved March 20 , 2016 .
</item 6>

<item 7>
Most Wanted (TV pilot)
^ Jump up to : `` Marvel 's Most Wanted '' . MckennaGrace.com . May 3 , 2016 . Archived from the original on July 14 , 2017 . Retrieved July 14 , 2017 . ^ Jump
up to : Frederick , Brittany . `` Who directed Spider - Man : Homecoming ? '' . Fansided . Archived from the original on July 14 , 2017 . Retrieved July 14 ,
2017 . ^ Jump up to : Walsh , S.M. ( June 6 , 2016 ) . `` ' Stitchers ' Season 3 : Meet the Cast '' . Heavy.com . Archived from the original on July 14 , 2017 .
Retrieved July 14 , 2017 . Jump up ^ Daniels , Hunter ( July 23 , 2011 ) . `` Comic - Con : Marvel TV Announces Development Slate ; Includes Live - Action AKA
Jessica Jones ( Alias ) , The Hulk , Cloak And Dagger , Mockingbird , Animated Hulk And The Agents Of S.M.A.S.H. , More '' .
</item 7>

<item 8>
Most Wanted (TV pilot)
The series would have been based on storylines occurring at the end of that second season , but ABC passed on the project . In August 2015 , the series was
reinvented as Most Wanted and received a pilot order from ABC . The cast was filled out in early 2016 , and production was completed by late March . The episode
was co-produced by ABC Studios and Marvel Television . In May 2016 , ABC announced that it would not pick up the pilot to series , and would look to develop
other series with Marvel instead . Contents ( hide ) 1 Premise 2 Cast and characters 2.1 Main 2.2 Guests 3 Production 3.1 Development 3.2 Writing 3.3 Casting
3.4 Filming 3.5 Marvel Cinematic Universe tie - ins 4 No pick - up and future 5 References 6 External links Premise ( edit ) Ex-spies and ex-spouses Bobbi Morse
and Lance Hunter are on the run trying to uncover a conspiracy against them . With no help from S.H.I.E.L.D.
</item 8>

<item 9>
Most Wanted (TV pilot)
At that time , the producers were prepping `` if - come '' orders , `` deals with writers that will come into effect in the event a pilot is picked ( up ) to
series '' . On May 12 , 2016 , the pilot was passed on by ABC , with Lee 's successor Channing Dungey explaining that Most Wanted `` did not feel as strong as
some of the other pilots that we shot . We talked about it with Marvel and we all came to an agreement that we want to figure out what the next show is that we
do together , is something that we all feel is as creatively strong as it can be . '' However , Dungey said that ABC was open to the series being shopped to
another network if Marvel found `` another home '' for it . Loeb also acknowledged this possibility , but said that the series was designed `` to do a very
specific thing '' alongside S.H.I.E.L.D. on ABC . References ( edit ) ^ Jump up to : `` 2016 ABC Pilots '' . Deadline.com . February 22 , 2016 . Archived from
the original on February 23 , 2016 .
</item 9>

<item 10>
Most Wanted (TV pilot)
^ Jump up to : Schwartz , Terri ( March 22 , 2016 ) . `` Agents of SHIELD EP Jeff Bell on Mockingbird and Hunter Departure and Marvel 's Most Wanted '' . IGN .
Archived from the original on March 23 , 2016 . Retrieved March 23 , 2016 . Jump up ^ Schwartz , Terri ( January 9 , 2016 ) . `` Agents of SHIELD Showrunners
Explain Marvel 's Most Wanted Spinoff Plan '' . IGN . Archived from the original on January 10 , 2016 . Retrieved January 10 , 2016 . ^ Jump up to : Lovett ,
Jamie ( October 27 , 2015 ) . `` EXCLUSIVE Agents Of S.H.I.E.L.D. EP Talks Secret Warriors , Civil War , Last Week 's Shocking Death & More '' . ComicBook.com .
Archived from the original on October 27 , 2015 . Retrieved October 28 , 2015 . ^ Jump up to : Bucksbaum , Sydney ( March 18 , 2016 ) .
</item 10>
=== END ITEM id=78f9daa6a8364491 ===

=== ITEM id=f7400859071a5911 ===
Gold sentences and facts:
Sentence 0: Songwriter ( s ) Kirsty MacColl / Philip Rambow
  Fact 0.0: The songwriters of the song are Kirsty MacColl and Philip Rambow.
Sentence 1: `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 )
  Fact 1.0: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is a single by Kirsty MacColl.
  Fact 1.1: 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' is from the album Desperate Character.
  Fact 1.2: The single's A-side is 'There's a Guy Works Down the Chip Shop Swears He's Elvis'.
  Fact 1.3: The single's B-side is 'Hard To Believe There's a Guy ... (country version)'.
  Fact 1.4: The single was released in May 1981.
  Fact 1.5: The single's format is 7-inch vinyl.
  Fact 1.6: The single's genre is country pop.
  Fact 1.7: The single's label is Polydor.
  Fact 1.8: The single's songwriters are Kirsty MacColl and Philip Rambow.
  Fact 1.9: The single's producer is Barry Farmer.
  Fact 1.10: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981).
  Fact 1.11: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop, Swears He's Elvis' (1981).
  Fact 1.12: The Kirsty MacColl singles chronology includes 'See That Girl' (1981).
  Fact 1.13: The Kirsty MacColl singles chronology includes 'Keep Your Hands Off My Baby' (1981) a second time.
  Fact 1.14: The Kirsty MacColl singles chronology includes 'There's a Guy Works Down the Chip Shop Swears He's Elvis' (1981).
  Fact 1.15: The Kirsty MacColl singles chronology includes 'See That Girl' (1981) a second time.

Retrieved items:
<item 1>
Sacrifice (song)
Jump up ^ `` UK certifications , database '' . Bpi . Retrieved 9 September 2008 . External links ( edit ) Lyrics of this song at MetroLyrics Elton John songs
1960s and 1970s `` Skyline Pigeon '' ( 1968 ) `` I 've Been Loving You '' ( 1968 ) `` Lady Samantha '' ( 1969 ) `` It 's Me That You Need '' ( 1969 ) `` Border
Song '' ( 1970 ) `` Rock and Roll Madonna '' ( 1970 ) `` From Denver to L.A. '' ( 1970 ) `` Take Me to the Pilot '' ( 1970 ) `` Your Song '' ( 1970 ) `` Friends
'' ( 1971 ) `` Levon '' ( 1971 ) `` Tiny Dancer '' ( 1972 ) `` Rocket Man '' ( 1972 ) `` Honky Cat '' ( 1972 ) `` I Think I 'm Going to Kill Myself '' ( 1972 )
`` Mona Lisas and Mad Hatters '' ( 1972 ) `` Crocodile Rock '' ( 1972 ) `` Elderberry Wine '' (
</item 1>

<item 2>
There's a Guy Works Down the Chip Shop Swears He's Elvis
There 's a Guy Works Down the Chip Shop Swears He 's Elvis - wikipedia There 's a Guy Works Down the Chip Shop Swears He 's Elvis Jump to : navigation , search
This article relies too much on references to primary sources . Please improve this by adding secondary or tertiary sources . ( March 2008 ) ( Learn how and
when to remove this template message ) `` There 's a Guy Works Down the Chip Shop , Swears He 's Elvis '' Single by Kirsty MacColl from the album Desperate
Character A-side `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' B - side `` Hard To Believe There 's a Guy ... ( country version ) '' Released
May 1981 Format 7 '' Vinyl Genre Country pop Label Polydor Songwriter ( s ) Kirsty MacColl / Philip Rambow Producer ( s ) Barry Farmer Kirsty MacColl singles
chronology `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop ,
</item 2>

<item 3>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Track listing ( edit ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' ( K. MacColl / P. Rambow ) `` Hard to Believe '' ( K. MacColl ) `` There
's a Guy Works Down the Chip Shop Swears He 's Elvis ( country version ) '' ( K. MacColl / P. Rambow ) References ( edit ) Jump up ^ `` There 's a Guy Works
Down the Chip Shop Swears He 's Elvis - Kirsty MacColl '' . Retrieved 2014 - 10 - 30 . Jump up ^ `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis 7
'' single - Kirsty MacColl `` . Retrieved 2014 - 10 - 30 .
</item 3>

<item 4>
All I Want for Christmas Is My Two Front Teeth
The song was introduced in 1948 by singing group The Satisfiers on Perry Como 's radio show , and originally recorded by Spike Jones & His City Slickers on
December 6 , 1947 , with lead vocal by George Rock . That version reached the top of the pop charts in 1948 , and again in 1949 . The song has also been
recorded by numerous other singers and performers , including Ray Stevens , George Strait , Danny Kaye with The Andrews Sisters , Urbie Green , The Platters ,
Dread Zeppelin , The Kelly Family , Nat King Cole ( reportedly Gardner 's favorite version ) , The Chipmunks , the Hampton String Quartet , The Three Stooges
and Count von Count of Sesame Street . In the versions by Spike , The Chipmunks ( Theodore on lead vocals ) , and Ray Stevens , the song ends with the performer
shouting the words `` Happy New Year ! ! ! '' Homer and Jethro parodied the song as `` All I Want for Christmas Is My Upper Plate '' , on their 1968 album Cool
Crazy Christmas .
</item 4>

<item 5>
Neil Diamond
`` I 'm a Believer '' became a gold record within two days of its release , and stayed at the top of the charts for seven weeks , making it the Popular Music
Song of the Year in 1966 . `` And the Grass Wo n't Pay No Mind '' brought covers from Elvis Presley ( who also interpreted `` Sweet Caroline '' ) and Mark
Lindsay , former lead singer for Paul Revere & the Raiders . Other notable artists who recorded his early songs were the English hard - rock band Deep Purple ,
Lulu , and Cliff Richard . In 1966 , Diamond signed a deal with Bert Berns 's Bang Records , then a subsidiary of Atlantic . His first release on that label ,
`` Solitary Man '' , became his first true hit as a solo artist . Diamond later followed with `` Cherry , Cherry '' and `` Kentucky Woman '' . His early
concerts saw him as a `` special guest '' of , or opening for , everyone from Herman 's Hermits to , on one occasion , the Who .
</item 5>

<item 6>
The Rolling Stones
Jump up ^ Haslam 2015 , p. 91 . Jump up ^ Wyman 2002 , pp. 65 . Jump up ^ Wyman 2002 , pp. 80 -- 83 . Jump up ^ Fricke , David ( 17 April 2008 ) . `` Blues
Brothers '' . Rolling Stone . Archived from the original on 5 April 2008 . Jump up ^ Gilliland 1969 , show 30 , track 2 . Jump up ^ `` Yes , The Beatles Once
Wrote a Song for the Rolling Stones '' . Ultimate Classic Rock . Archived from the original on 9 July 2017 . Retrieved 15 August 2017 . Jump up ^ `` How The
Beatles ended up writing `` I Wanna Be Your Man '' for The Rolling Stones `` . The Beatles . Archived from the original on 15 August 2017 . Retrieved 15 August
2017 . ^ Jump up to : `` UK Charts -- The Rolling Stones '' . Official Charts Company . Archived from the original on 20 October 2014 .
</item 6>

<item 7>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Retrieved 2014 - 10 - 30 . Jump up ^ Information at Svensk mediedatabas External links ( edit ) freeworld - Kirsty MacColl fansite Kirsty MacColl Studio albums
Desperate Character Kirsty MacColl Kite Electric Landlady Titanic Days Tropical Brainstorm Compilations Galore Notable singles `` They Do n't Know '' `` Keep
Your Hands Off My Baby '' `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' `` You Still Believe in Me '' `` Terry '' `` A New England '' `` Days
'' `` Do n't Come The Cowboy With Me Sonny Jim ! ''
</item 7>

<item 8>
There's a Guy Works Down the Chip Shop Swears He's Elvis
Swears He 's Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` Keep Your Hands Off My Baby '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's
Elvis '' ( 1981 ) `` See That Girl '' ( 1981 ) `` There 's a Guy Works Down the Chip Shop Swears He 's Elvis '' was Kirsty MacColl 's first hit single ,
charting in the UK at # 14 in 1981 , and remaining in the charts for nine weeks . It reached # 9 in the Irish charts . A promotional version was released in the
USA , which instead referenced a truck stop as opposed to the very British chip shop , but this was only available to DJs and did not get a full release .
Norwegian singer Elisabeth Andreasson covered the song on her 1981 country album Angel of the Morning , with lyrics in Swedish by Hasse Olsson as `` Killen ner
' på Konsum svär att han är Elvis '' ( `` The guy down Konsum '' swears he 's Elvis ) .
</item 8>

<item 9>
There's a Guy Works Down the Chip Shop Swears He's Elvis
`` Walking Down Madison '' `` Soho Square '' With The Pogues `` Fairytale of New York '' Related articles Ewan MacColl Steve Lillywhite Retrieved from ``
https://en.wikipedia.org/w/index.php?title=There%27s_a_Guy_Works_Down_the_Chip_Shop_Swears_He%27s_Elvis&oldid=799608311 '' Categories : 1981 singles 1981 songs
Elisabeth Andreassen songs Kirsty MacColl songs Songs written by Kirsty MacColl Songs about Elvis Presley Polydor Records singles Hidden categories : Articles
lacking reliable references from March 2008 All articles lacking reliable references Talk About Wikipedia Svenska Edit links This page was last edited on 8
September 2017 , at 19 : 12 . About Wikipedia
</item 9>

<item 10>
Neil Diamond
He did songwriting wherever he could , including on buses , and used an upright piano above the Birdland Club in New York City . One of the causes of this early
nomadic life as a songwriter was due to his songs having too many words : `` I 'd spent a lot of time on lyrics , and they were looking for hooks , and I did
n't really understand the nature of that , '' he says . During those lean years , he was only able to sell about one song a week , barely enough to survive on .
He found himself only earning enough to spend 35 cents a day on food ( US $3 in 2017 dollars ) . However , the privacy he had above the Birdland Club allowed
him to focus on writing without distractions ; as he explained , `` Something new began to happen . I was n't under the gun , and suddenly interesting songs
began to happen , songs that had things none of the others did . '' Among them were `` Cherry , Cherry '' and `` Solitary Man '' . `` Solitary Man '' was the
first record that Diamond recorded in his own name that made the charts .
</item 10>
=== END ITEM id=f7400859071a5911 ===

=== ITEM id=f805bc78dfb402d7 ===
Gold sentences and facts:
Sentence 0: A muscle relaxant is a drug that affects skeletal muscle function and decreases the muscle tone . It may be used to alleviate symptoms such as muscle spasms , pain , and hyperreflexia . The term `` muscle relaxant '' is used to refer to two major therapeutic groups : neuromuscular blockers and spasmolytics . Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate and have no central nervous system ( CNS ) activity . They are often used during surgical procedures and in intensive care and emergency medicine to cause temporary paralysis . Spasmolytics , also known as `` centrally acting '' muscle relaxants , are used to alleviate musculoskeletal pain and spasms and to reduce spasticity in a variety of neurological conditions . While both neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants , the term is commonly used to refer to spasmolytics only .
  Fact 0.0: A muscle relaxant is a drug that affects skeletal muscle function.
  Fact 0.1: A muscle relaxant decreases muscle tone.
  Fact 0.2: A muscle relaxant may be used to alleviate muscle spasms.
  Fact 0.3: A muscle relaxant may be used to alleviate pain.
  Fact 0.4: A muscle relaxant may be used to alleviate hyperreflexia.
  Fact 0.5: The term 'muscle relaxant' is used to refer to two major therapeutic groups.
  Fact 0.6: The two major therapeutic groups referred to by 'muscle relaxant' are neuromuscular blockers and spasmolytics.
  Fact 0.7: Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate.
  Fact 0.8: Neuromuscular blockers have no central nervous system (CNS) activity.
  Fact 0.9: Neuromuscular blockers are often used during surgical procedures.
  Fact 0.10: Neuromuscular blockers are often used in intensive care and emergency medicine.
  Fact 0.11: Neuromuscular blockers are used to cause temporary paralysis.
  Fact 0.12: Spasmolytics are also known as 'centrally acting' muscle relaxants.
  Fact 0.13: Spasmolytics are used to alleviate musculoskeletal pain and spasms.
  Fact 0.14: Spasmolytics are used to reduce spasticity in a variety of neurological conditions.
  Fact 0.15: Neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants.
  Fact 0.16: The term 'muscle relaxant' is commonly used to refer to spasmolytics only.

Retrieved items:
<item 1>
Graph query row:
entity: Spasmolytics
relationship: RELATED_TO
clinical_use: neck pain
</item 1>

<item 2>
Graph query row:
entity: Spasmolytics
relationship: RELATED_TO
clinical_use: myofascial pain syndrome
</item 2>

<item 3>
Graph query row:
entity: Spasmolytics
relationship: RELATED_TO
clinical_use: Muscle spasms
</item 3>

<item 4>
Graph query row:
entity: Spasmolytics
relationship: RELATED_TO
clinical_use: low back pain
</item 4>

<item 5>
Graph query row:
entity: Spasmolytics
relationship: RELATED_TO
clinical_use: tension headaches
</item 5>

<item 6>
Muscle relaxant
dantrolene ) , so this name is inaccurate . Most sources still use the term `` centrally acting muscle relaxant '' . According to MeSH , dantrolene is usually
classified as a centrally acting muscle relaxant . The World Health Organization , in its ATC , uses the term `` centrally acting agents '' , but adds a
distinct category of `` directly acting agents '' , for dantrolene . Use of this terminology dates back to at least 1973 . The term `` spasmolytic '' is also
considered a synonym for antispasmodic . Clinical use ( edit ) Spasmolytics such as carisoprodol , cyclobenzaprine , metaxalone , and methocarbamol are commonly
prescribed for low back pain or neck pain , fibromyalgia , tension headaches and myofascial pain syndrome . However , they are not recommended as first - line
agents ; in acute low back pain , they are not more effective than paracetamol or nonsteroidal anti-inflammatory drugs ( NSAIDs ) , and in fibromyalgia they are
not more effective than antidepressants .
</item 6>

<item 7>
Graph query row:
entity: Spasmolytics
relationship: RELATED_TO
clinical_use: Musculoskeletal pain
</item 7>

<item 8>
Muscle relaxant
Muscle relaxant - wikipedia Muscle relaxant Jump to : navigation , search This article is about skeletal muscle relaxants . For information on smooth muscle
relaxants , see Antispasmodic . A muscle relaxant is a drug that affects skeletal muscle function and decreases the muscle tone . It may be used to alleviate
symptoms such as muscle spasms , pain , and hyperreflexia . The term `` muscle relaxant '' is used to refer to two major therapeutic groups : neuromuscular
blockers and spasmolytics . Neuromuscular blockers act by interfering with transmission at the neuromuscular end plate and have no central nervous system ( CNS
) activity . They are often used during surgical procedures and in intensive care and emergency medicine to cause temporary paralysis . Spasmolytics , also
known as `` centrally acting '' muscle relaxants , are used to alleviate musculoskeletal pain and spasms and to reduce spasticity in a variety of neurological
conditions . While both neuromuscular blockers and spasmolytics are often grouped together as muscle relaxants , the term is commonly used to refer to
spasmolytics only .
</item 8>

<item 9>
Graph query row:
entity: Spasmolytics
relationship: RELATED_TO
clinical_use: Spasticity
</item 9>

<item 10>
Graph query row:
entity: Spasmolytics
relationship: RELATED_TO
clinical_use: fibromyalgia
</item 10>
=== END ITEM id=f805bc78dfb402d7 ===

=== ITEM id=207511fb2a155d97 ===
Gold sentences and facts:
Sentence 0: `` Ca n't Get You Out of My Head '' is a song recorded by Australian singer Kylie Minogue for her eighth studio album , titled Fever , which she released in 2001 . The song was released in Australia by Parlophone as the lead single from the album on 8 September 2001 . It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed , and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's obsession towards her lover . The song is famous for its `` la la la '' hook .
  Fact 0.0: 'Can't Get You Out of My Head' is a song.
  Fact 0.1: 'Can't Get You Out of My Head' is recorded by Australian singer Kylie Minogue.
  Fact 0.2: 'Can't Get You Out of My Head' was recorded for Kylie Minogue's eighth studio album.
  Fact 0.3: The eighth studio album is titled Fever.
  Fact 0.4: Kylie Minogue released Fever in 2001.
  Fact 0.5: 'Can't Get You Out of My Head' was released in Australia by Parlophone.
  Fact 0.6: 'Can't Get You Out of My Head' was released in Australia as the lead single from Fever.
  Fact 0.7: The Australian release of 'Can't Get You Out of My Head' was on 8 September 2001.
  Fact 0.8: 'Can't Get You Out of My Head' was released in the United Kingdom on 17 September 2001.
  Fact 0.9: 'Can't Get You Out of My Head' was released in the United States as a single on 18 February 2002.
  Fact 0.10: 'Can't Get You Out of My Head' was jointly written by Cathy Dennis and Rob Davis.
  Fact 0.11: 'Can't Get You Out of My Head' was jointly composed by Cathy Dennis and Rob Davis.
  Fact 0.12: 'Can't Get You Out of My Head' was jointly produced by Cathy Dennis and Rob Davis.
  Fact 0.13: 'Can't Get You Out of My Head' is a midtempo dance-pop song.
  Fact 0.14: 'Can't Get You Out of My Head' lyrically details its narrator's obsession towards her lover.
  Fact 0.15: The song is famous for its 'la la la' hook.

Retrieved items:
<item 1>
“Can’t Get You Out of My Head” is a synth-pop/dance-pop single by Australian singer Kylie Minogue, released by Parlophone on 8 September 2001 as the lead single
from her eighth album, *Fever*. Recorded in London, it runs 3:50 and was written and produced by Cathy Dennis and Rob Davis. Its B-sides are “Boy” and
“Rendezvous at Sunset.”
</item 1>

<item 2>
“Can’t Get You Out of My Head” depicts an unnamed, possibly unattainable object of obsession. Critics heard a darker, potentially destructive desire beneath its
simple lyrics and restrained vocals, contrasting it with the optimism of Minogue’s “I Should Be So Lucky”; the singer recognizes the need for self-control. Its
hypnotic “la la la” refrain and repetitive synth beat helped establish Minogue’s relevance and a cooler, machine-like sexuality. A 2013 Manchester Science
Festival survey named
</item 2>

<item 3>
“Can’t Get You Out of My Head” is a robotic, midtempo dance-pop song at 126 BPM, famous for its “la la la” hook. Its music, apart from the guitar part, was
programmed on a Korg Triton workstation via MIDI. Dennis said Kylie was the right artist for the song and praised her performance and video.
</item 3>

<item 4>
Can't Get You Out of My Head
The song was recorded using Cubase music software , which Davis ran on his Mac computer . Davis began playing an acoustic guitar and ran a 125 beats per minute
drum loop , on which Dennis , herself a singer who had enjoyed chart success in the United States , began singing the line `` I just ca n't get you out of my
head '' in the key of D minor . After three and a half hours , the demo was recorded and the vocals were laid afterwards . Dennis called their recording setup
for the song `` the most primitive set - up you could imagine ! Different producers work in different ways . But it 's good to be reminded you do n't have to be
reliant on equipment . A song is about melody and lyrics and being able to take something away in your memory that is going to haunt you . '' She also regarded
their production as a `` very natural and fluid process , '' saying : `` We know how hard we work sometimes to write songs and then spend months picking them to
pieces , but this was the easiest process , the chemicals were all happy and working together . ''
</item 4>

<item 5>
Can't Get You Out of My Head
Through the lyrics of the song , its singer expresses an obsession with an anonymous figure . Dorian Lynskey from The Guardian termed the song a `` mystery ''
as Minogue never reveals the identity of her object of infatuation . The critic suggested that the person Minogue is referring to is either `` a partner , an
evasive one - night stand or someone who does n't know she exists . '' Writing for the same newspaper , Everett True identified a `` darker element '' in the
simple lyrics and felt this sentiment was echoed in Minogue 's restrained vocals . Further , True emphasised that while Minogue 's 1987 single `` I Should Be So
Lucky '' had presented an optimistic romantic future , `` Ca n't Get You Out of My Head '' focuses on an `` unhealthy '' and potentially destructive obsession .
He also noted that in the former song , Minogue played `` the wide - eyed ingénue with alacrity , '' but in the latter song she is aware of the harmful nature
of her infatuation , calling it a `` desire that is wholly dependent on her own self - control . ''
</item 5>

<item 6>
Can't Get You Out of My Head
Ca n't Get You Out of My Head - Wikipedia Ca n't Get You Out of My Head Jump to : navigation , search `` Ca n't Get You Out of My Head '' Single cover used for
CD # 02 and digital releases of the song .
</item 6>

<item 7>
Released in the UK on 17 September 2001 and in the US on 18 February 2002, Kylie Minogue’s “Can’t Get You Out of My Head” was written and produced by Cathy
Dennis and Rob Davis. The midtempo dance-pop song, known for its “la la la” hook, describes obsessive love. It topped charts across Europe, Australia, Canada
and New Zealand, reportedly reaching No. 1 in 40 countries, and peaked at No.
</item 7>

<item 8>
Can't Get You Out of My Head
The music , excluding the guitar part , was programmed using Korg Triton workstation via MIDI . Dennis later remarked : `` Even though Kylie was n't the first
artist to be offered the song , I do n't believe it was meant to go to anyone other than Kylie , and I do n't believe anyone else would have done the incredible
job she did with it , with the video , looking super-hot ! '' Composition ( edit ) `` Ca n't Get You Out of My Head '' A 21 - second sample of `` Ca n't Get You
Out of My Head '' containing the hook and part of the chorus of the song . The song is known for its `` la la la '' hook . Problems playing this file ? See
media help . `` Ca n't Get You Out of My Head '' is a `` robotic '' midtempo dance - pop song , with a tempo of 126 beats per minute .
</item 8>

<item 9>
Can't Get You Out of My Head
My A&R at the time , Miles Leonard and Jamie Nelson , said , `` We 've got something . Come into the office . We 've got to play you something . '' So I went to
the EMI office , and I had the same reaction you did , about 20 seconds in . I could n't even fathom what I was hearing . It just ... did something . I was
beside myself . Then at the end of the song , panic set in . I was saying `` Are you sure we 've got this song ? Do n't tell me that we do n't ! Is it secured ?
Can we have it ? '' And we did ! And that kick - started a whole different phase in my career . -- Minogue , about her reaction to hearing the demo of the song
. `` Ca n't Get You Out of My Head '' was jointly written , composed , and produced by Cathy Dennis and Rob Davis . Dennis and Davis had been brought together
by British artist manager Simon Fuller , who wanted the duo to come up with a song for British pop group S Club 7 .
</item 9>

<item 10>
Can't Get You Out of My Head
It was released on 17 September 2001 in the United Kingdom . In the United States , the single was released on 18 February 2002 . Jointly written , composed ,
and produced by Cathy Dennis and Rob Davis , `` Ca n't Get You Out of My Head '' is a midtempo dance - pop song which lyrically details its narrator 's
obsession towards her lover . The song is famous for its `` la la la '' hook . In addition to positive reviews from music critics , `` Ca n't Get You Out of My
Head '' found commercial success on a large scale . It peaked at number one on the charts of Austria , Belgium , France , Germany , Italy , Poland , Switzerland
, the United Kingdom , and every other European country excluding Finland . It also topped the charts of Minogue 's native country Australia , Canada and New
Zealand . In the United States , the song peaked at number seven on the US Billboard Hot 100 chart , becoming Minogue 's biggest hit in the region since `` The
Loco - Motion '' . `` Ca n't Get You Out of My Head '' reportedly reached number one in 40 countries across the globe .
</item 10>
=== END ITEM id=207511fb2a155d97 ===

=== ITEM id=244c92a490d13b13 ===
Gold sentences and facts:
Sentence 0: At the Council , following advice offered by Simon Peter ( Acts 15 : 7 -- 11 and Acts 15 : 14 ) , Barnabas and Paul gave an account of their ministry among the gentiles ( Acts 15 : 12 ) , and the apostle James quoted from the words of the prophet Amos ( Acts 15 : 16 -- 17 , quoting Amos 9 : 11 -- 12 ) . James added his own words to the quotation : `` Known to God from eternity are all His works '' and then submitted a proposal , which was accepted by the Church and became known as the Apostolic Decree :
  Fact 0.0: Simon Peter offered advice at the Council, recorded in Acts 15:7-11 and Acts 15:14.
  Fact 0.1: Barnabas and Paul gave an account of their ministry among the gentiles at the Council, recorded in Acts 15:12.
  Fact 0.2: The account given by Barnabas and Paul followed advice offered by Simon Peter.
  Fact 0.3: The apostle James quoted from the words of the prophet Amos, recorded in Acts 15:16-17.
  Fact 0.4: The apostle James quoted Amos 9:11-12.
  Fact 0.5: James added his own words to the quotation: 'Known to God from eternity are all His works'.
  Fact 0.6: James submitted a proposal.
  Fact 0.7: The proposal was accepted by the Church.
  Fact 0.8: The proposal became known as the Apostolic Decree.

Retrieved items:
<item 1>
Council of Jerusalem
, Paget , `` Jewish Christianity '' , in Horbury , et al. , `` The Cambridge History of Judaism : The Early Roman Period '' , volume 3 , p. 744 ( 2008 ) .
Cambridge University Press . Jump up ^ `` However , numerous scholars have challenged the historicity of the Jerusalem Council as related by Acts , Paul 's
presence there in the manner that Luke describes , the issue of idol - food being thrust on Paul 's Gentile mission , and the historical reliability of Acts in
general . '' , Fotopolous , `` Food Offered to Idols in Roman Corinth : a socio - rhetorical reconsideration '' , pp. 181 -- 182 ( 2003 ) . Mohr Siebeck . Jump
up ^ `` Sahlin rejects the historicity of Acts completely ( Der Messias und das Gottesvolk ( 1945 ) ) . Haenchen 's view is that the Apostolic Council `` is an
imaginary construction answering to no historical reality '' ( The Acts of the Apostles ( Engtr 1971 ) , p. 463 ) .
</item 1>

<item 2>
Council of Jerusalem
The council decided that Gentile converts to Christianity were not obligated to keep most of the Law of Moses , including the rules concerning circumcision of
males . The Council did , however , retain the prohibitions on eating blood , meat containing blood , and meat of animals not properly slain , and on
fornication and idolatry , sometimes referred to as the Apostolic Decree or Jerusalem Quadrilateral . Accounts of the council are found in Acts of the Apostles
chapter 15 ( in two different forms , the Alexandrian and Western versions ) and also possibly in Paul 's letter to the Galatians chapter 2 . Some scholars
dispute that Galatians 2 is about the Council of Jerusalem ( notably because Galatians 2 describes a private meeting ) while other scholars dispute the
historical reliability of the Acts of the Apostles . Contents ( hide ) 1 Historical background 2 Issues and outcome 3 Historicity 4 Interpreting the Council 's
decision 4.1 Modern interpretation 4.1.
</item 2>

<item 3>
Council of Jerusalem
No one will pretend that the disciplinary enactments of any council , even though it be one of the undisputed Ecumenical Synods , can be of greater and more
unchanging force than the decree of that first council , held by the Holy Apostles at Jerusalem , and the fact that its decree has been obsolete for centuries
in the West is proof that even Ecumenical canons may be of only temporary utility and may be repealed by disuse , like other laws . '' ^ Hillel the Elder when
asked by a Gentile to teach the whole Torah while standing on one foot cited the negative form of the Golden Rule , also cited in Tobit 4 : 15 . Jesus in
Matthew 7 : 12 , part of the Sermon on the Mount , cited the positive form as summary of the `` Law and Prophets . '' ^ Whether or not Galatians 2 : 1 -- 10 is
a record of the Council of Jerusalem or a different event is not agreed . Paul writes of laying his gospel before the others `` privately , '' not as in a
Council .
</item 3>

<item 4>
Council of Jerusalem
^ There are two major versions of Acts : Alexandrian and Western ; with preference generally given to the Alexandrian , see Bruce Metzger 's Textual Commentary
on the Greek New Testament which has for the Western 15 : 2 , `` for Paul spoke maintaining firmly that they should stay as they were when converted ; but those
who had come from Jerusalem ordered them , Paul and Barnabas and certain others , to go up to Jerusalem to the apostles and elders that they might be judged
before them about this question . '' ^ According to Bruce Metzger 's Textual Commentary on the Greek New Testament : `` the Apostolic Decree ( 15.29 , 15.20 ,
21.25 ) ... contain many problems concerning text and exegesis '' ; `` it is possible ... ( fornication means ) marriage within the prohibited Levitical Degrees
( Leviticus 18 : 6 -- 18 ) , which the rabbis described as `` forbidden for porneia '' , or mixed marriages with pagans ( Numbers 25 : 1 ; also compare
2Corinthians 6.14 ) ,
</item 4>

<item 5>
Council of Jerusalem
463 ) . Dibelius ' view ( Studies in the Acts of the Apostles ( Engtr 1956 ) , pp. 93 -- 101 ) is that Luke 's treatment is literary - theological and can make
no claim to historical worth. `` , Mounce , `` Apostolic Council '' , in Bromiley ( ed . ) `` The International Standard Bible Encyclopedia '' , volume 1 , p.
200 ( rev . ed. 2001 ) . Wm . B. Eerdmans . Jump up ^ `` There is an increasing trend among scholars toward considering the Jerusalem Council as historical
event . An overwhelming majority identifies the reference to the Jerusalem Council in Acts 15 with Paul 's account in Gal. 2.1 -- 10 , and this accord is not
just limited to the historicity of the gathering alone but extends also to the authenticity of the arguments deriving from the Jerusalem church itself . '' ,
Philip , `` The Origins of Pauline Pneumatology : the Eschatological Bestowal of the Spirit '' , Wissenschaftliche Untersuchungen zum Neuen Testament 2 , Reihe
, p. 205 ( 2005 ) .
</item 5>

<item 6>
Council of Jerusalem
Authorship and development ( show ) Authorship Dating Hebrew canon Old Testament canon New Testament canon Documentary hypothesis Mosaic authorship Pauline
epistles Petrine epistles Johannine works Translations and manuscripts ( show ) Samaritan Torah Dead Sea scrolls Masoretic Text Targumim Peshitta Septuagint
Vulgate Gothic Bible Vetus Latina Luther Bible English Bibles Biblical studies ( show ) Archeology Artifacts Dating Historicity Internal consistency People
Places Names Novum Testamentum Graece Documentary hypothesis Synoptic problem NT textual categories Biblical criticism Historical Textual Source Form Redaction
Canonical Interpretation ( show ) Hermeneutics Pesher Midrash Pardes Allegorical interpretation Literalism Prophecy Inspiration Perspectives ( show ) Gnostic
Islamic Qur'anic Muhammad Inerrancy Infallibility Criticism of the Bible Outline of Bible - related topics Bible book Bible portal The Council of Jerusalem or
Apostolic Council was held in Jerusalem around AD 50 . It is unique among the ancient pre-ecumenical councils in that it is considered by Catholics and Orthodox
to be a prototype and forerunner of the later ecumenical councils and a key part of Christian ethics .
</item 6>

<item 7>
Council of Jerusalem
At the Council , following advice offered by Simon Peter ( Acts 15 : 7 -- 11 and Acts 15 : 14 ) , Barnabas and Paul gave an account of their ministry among the
gentiles ( Acts 15 : 12 ) , and the apostle James quoted from the words of the prophet Amos ( Acts 15 : 16 -- 17 , quoting Amos 9 : 11 -- 12 ) . James added his
own words to the quotation : `` Known to God from eternity are all His works '' and then submitted a proposal , which was accepted by the Church and became
known as the Apostolic Decree : It is my judgment , therefore , that we should not make it difficult for the Gentiles who are turning to God . Instead we should
write to them , telling them to abstain from food polluted by idols , from sexual immorality , from the meat of strangled animals and from blood . For the law
of Moses has been preached in every city from the earliest times and is read in the synagogues on every Sabbath .
</item 7>

<item 8>
Council of Jerusalem
In the view of some scholars , the reminder of James and the elders here is an expression of concern that Paul was not fully teaching the decision of the
Jerusalem Council 's letter to Gentiles , particularly in regard to non-strangled kosher meat , which contrasts with Paul 's advice to Gentiles in Corinth , to
`` eat whatever is sold in the meat markets '' ( 1 Corinthians 10 : 25 ) . Historicity ( edit ) See also : Historical reliability of the Acts of the Apostles
The description of the Apostolic Council in Acts 15 , generally considered the same event described in Galatians 2 , is considered by some scholars to be
contradictory to the Galatians account . The historicity of Luke 's account has been challenged , and was rejected completely by some scholars in the mid to
late 20th century . However , more recent scholarship inclines towards treating the Jerusalem Council and its rulings as a historical event , though this is
sometimes expressed with caution .
</item 8>

<item 9>
Council of Jerusalem
This is the reason why this Council was called by Jesus ' apostle James , who , according to Paul 's account in Galatians 2 : 10 concluded that Paul would not
have to demand his men to become circumcised , and according to Luke 's account in Acts 15 : 13 -- 21 , James not only permitted uncircumcised men to remain in
the group , but he said in closing the Council ( Acts 15 : 19 ) `` We should not trouble the Gentiles who are turning to God . '' As for Orthodox Judaism today
, the rabbis do not believe that Gentiles need to be circumcised , unless they wish to convert to Judaism , which is discouraged . Instead , Gentiles need only
follow the Seven Laws of Noah to be assured of a place in the `` World to Come '' .
</item 9>

<item 10>
Council of Jerusalem
70 ) . Interpreting the Council 's decision ( edit ) See also : Paul the Apostle and Judaism and Early Christianity Part of a series on Jewish Christianity
Figures Jesus John the Baptist Simon Peter Twelve Apostles James , brother of Jesus Simeon of Jerusalem Jude Paul Ancient groups Ebionites Elcesaites Nazarenes
Pejoratives Judaizers Legalists Pauline Christianity Recent groups Hebrew Christian movement Messianic Judaism Saint Thomas Christians Adversity Split of
Christianity and Judaism Paul and Judaism Marcionism Christian anti-semitism Constantine Writings Gospel of Matthew Epistle of James Clementine Didache Liturgy
of St James Book of Elchasai Jewish -- Christian gospels Gospel of the Ebionites Gospel of the Hebrews Gospel of the Nazarenes Issues Aramaic of Jesus Yeshua (
name ) Council of Jerusalem Expounding of the Law Sabbath Quartodecimanism Noahide laws James 's Apostolic Decree was that most Mosaic law , including the
requirement for circumcision of males , was not obligatory for Gentile converts , possibly in order to make it easier for them to join the movement .
</item 10>
=== END ITEM id=244c92a490d13b13 ===

=== ITEM id=31c5b9985739f731 ===
Gold sentences and facts:
Sentence 0: In healthy adults , there are two normal heart sounds , often described as a lub and a dub ( or dup ) , that occur in sequence with each heartbeat . These are the first heart sound ( S ) and second heart sound ( S ) , produced by the closing of the atrioventricular valves and semilunar valves , respectively . In addition to these normal sounds , a variety of other sounds may be present including heart murmurs , adventitious sounds , and gallop rhythms S and S .
  Fact 0.0: In healthy adults there are two normal heart sounds.
  Fact 0.1: The two normal heart sounds are often described as a lub and a dub (or dup).
  Fact 0.2: The two normal heart sounds occur in sequence with each heartbeat.
  Fact 0.3: The first heart sound (S) is produced by the closing of the atrioventricular valves.
  Fact 0.4: The second heart sound (S) is produced by the closing of the semilunar valves.
  Fact 0.5: The closing of the atrioventricular valves and the semilunar valves produces the first and second heart sounds, respectively.
  Fact 0.6: In addition to the normal sounds, a variety of other sounds may be present.
  Fact 0.7: The other sounds that may be present include heart murmurs, adventitious sounds, and gallop rhythms labeled S and S.

Retrieved items:
<item 1>
Heart sounds
This is a characteristic scratching , creaking , high - pitched sound emanating from the rubbing of both layers of inflamed pericardium . It is the loudest in
systole , but can often be heard at the beginning and at the end of diastole . It is very dependent on body position and breathing , and changes from hour to
hour . Surface anatomy ( edit ) The aortic area , pulmonic area , tricuspid area and mitral area are areas on the surface of the chest where the heart is
auscultated . Heart sounds result from reverberation within the blood associated with the sudden block of flow reversal by the valves closing . Because of this
, auscultation to determine function of a valve is usually not performed at the position of the valve , but at the position to where the sound waves reverberate
.
</item 1>

<item 2>
Heart sounds
Aortic blood flow quickly reverses back toward the left ventricle , catching the pocket - like cusps of the aortic valve , and is stopped by aortic valve
closure . Similarly , as the pressure in the right ventricle falls below the pressure in the pulmonary artery , the pulmonary valve closes . The S sound results
from reverberation within the blood associated with the sudden block of flow reversal . Splitting of S2 , also known as physiological split , normally occurs
during inhalation because the decrease in intrathoracic pressure increases the time needed for pulmonary pressure to exceed that of the right ventricular
pressure . A widely split S2 can be associated with several different cardiovascular conditions , including left bundle branch block , pulmonary stenosis , and
atrial septal defect . Extra heart sounds ( edit ) The rarer extra heart sounds form gallop rhythms and are heard in both normal and abnormal situations .
</item 2>

<item 3>
Heart sounds
See also ( edit ) Pulse Physical examination Precordial examination Benign pediatric heart murmur Iambic pentameter , a metre in poetry which follows a similar
rhythm to the human heartbeat Pulsatile tinnitus - hearing a heartbeat sound in one or both ears Notes and references ( edit ) Jump up ^
https://depts.washington.edu/physdx/heart/tech2.html Jump up ^ https://depts.washington.edu/physdx/heart/tech2.html Jump up ^ `` thrill '' . Medline Plus
Medical Dictionary . Jump up ^ Bernstein , Lenny ( 2016 - 01 - 02 ) , `` Heart doctors are listening for clues to the future of their stethoscopes '' ,
Washington Post . External links ( edit ) University of Michigan Heart Sound and Murmur Library . Auscultation Assistant . UCLA Heart Sounds - Heart Murmurs .
practicalclinicalskills.com Physiology of the cardiovascular system Heart Cardiac output Cardiac cycle Cardiac output Heart rate Stroke volume Stroke volume End
- diastolic volume End - systolic volume Afterload Preload Frank -- Starling law Cardiac function curve Venous return curve Wiggers diagram Pressure volume
diagram Ultrasound Fractional
</item 3>

<item 4>
Heart sounds
Heart sounds - wikipedia Heart sounds Jump to : navigation , search Normal heart sounds Normal heart sounds as heard with a stethoscope Problems playing this
file ? See media help . 72 BPM Heart sound Front of thorax , showing surface relations of bones , lungs ( purple ) , pleura ( blue ) , and heart ( red outline )
. The locations of best auscultation for each heart valve are labeled with `` M '' , `` T '' , `` A '' , and `` P '' . First heart sound : caused by
atrioventricular valves - Mitral ( M ) and Tricuspid ( T ) . Second heart sound caused by semilunar valves -- Aortic ( A ) and Pulmonary / Pulmonic ( P ) .
Diagram showing relations of opened heart to front of thoracic wall . Ant . Anterior segment of tricuspid valve . A O. Aorta . A.P. Anterior papillary muscle .
In . Innominate artery . L.C.C. Left common carotid artery . L.S. Left subclavian artery . L.V. Left ventricle . P.A. Pulmonary artery .
</item 4>

<item 5>
Heart sounds
The sudden tensing of the chordae tendineae and the squeezing of the ventricles against closed semilunar valves , sends blood rushing back toward the atria ,
and the parachute - like valves catch the rush of blood in their leaflets causing the valve to snap shut . The S1 sound results from reverberation within the
blood associated with the sudden block of flow reversal by the valves . The delay of T1 even more than normally causes the split S1 which is heard in a right
bundle branch blockage . The second heart sound , or S , forms the `` dub '' of `` lub - dub '' and is composed of components A ( aortic valve closure ) and P (
pulmonary valve closure ) . Normally A precedes P especially during inspiration where a split of S can be heard . It is caused by the closure of the semilunar
valves ( the aortic valve and pulmonary valve ) at the end of ventricular systole and the beginning of ventricular diastole . As the left ventricle empties ,
its pressure falls below the pressure in the aorta .
</item 5>

<item 6>
Heart sounds
Main article : Third heart sound Rarely , there may be a third heart sound also called a protodiastolic gallop , ventricular gallop , or informally the ``
Kentucky '' gallop as an onomatopoeic reference to the rhythm and stress of S1 followed by S2 and S3 together ( S1 = Ken ; S2 = tuck ; S3 = y ) . `` lub - dub -
ta '' or `` slosh - ing - in '' If new , indicates heart failure or volume overload . It occurs at the beginning of diastole after S2 and is lower in pitch than
S1 or S2 as it is not of valvular origin . The third heart sound is benign in youth , some trained athletes , and sometimes in pregnancy but if it re-emerges
later in life it may signal cardiac problems , such as a failing left ventricle as in dilated congestive heart failure ( CHF ) . S3 is thought to be caused by
the oscillation of blood back and forth between the walls of the ventricles initiated by blood rushing in from the atria .
</item 6>

<item 7>
Heart sounds
The reason the third heart sound does not occur until the middle third of diastole is probably that during the early part of diastole , the ventricles are not
filled sufficiently to create enough tension for reverberation . It may also be a result of tensing of the chordae tendineae during rapid filling and expansion
of the ventricle . In other words , an S3 heart sound indicates increased volume of blood within the ventricle . An S3 heart sound is best heard with the bell -
side of the stethoscope ( used for lower frequency sounds ) . A left - sided S3 is best heard in the left lateral decubitus position and at the apex of the
heart , which is normally located in the 5th left intercostal space at the midclavicular line . A right - sided S3 is best heard at the lower - left sternal
border . The way to distinguish between a left and right - sided S3 is to observe whether it increases in intensity with inhalation or exhalation . A right -
sided S3 will increase on inhalation , while a left - sided S3 will increase on exhalation .
</item 7>

<item 8>
Heart sounds
At rapid heart rates , S3 and S4 may merge to produce a summation gallop , sometimes referred to as S7 . Atrial contraction must be present for production of an
S4 . It is absent in atrial fibrillation and in other rhythms in which atrial contraction does not precede ventricular contraction . Murmurs ( edit )
Auscultogram from normal and abnormal heart sounds . Heart murmurs are produced as a result of turbulent flow of blood strong enough to produce audible noise .
They are usually heard as a whooshing sound . The term murmur only refers to a sound believed to originate within blood flow through or near the heart ; rapid
blood velocity is necessary to produce a murmur . Most heart problems do not produce any murmur and most valve problems also do not produce an audible murmur .
Murmurs can be heard in many situations in adults without major congenital heart abnormalities : Regurgitation through the mitral valve is by far the most
commonly heard murmur , producing a pansystolic / holosystolic murmur which is sometimes fairly loud to a practiced ear , even though the volume of regurgitant
blood flow may be quite small .
</item 8>

<item 9>
Heart sounds
L.V. Left ventricle . P.A. Pulmonary artery . R.A. Right atrium . R.V. Right ventricle . V.S. Ventricular septum . Heart sounds are the noises generated by the
beating heart and the resultant flow of blood through it . Specifically , the sounds reflect the turbulence created when the heart valves snap shut . In cardiac
auscultation , an examiner may use a stethoscope to listen for these unique and distinct sounds that provide important auditory data regarding the condition of
the heart . In healthy adults , there are two normal heart sounds , often described as a lub and a dub ( or dup ) , that occur in sequence with each heartbeat .
These are the first heart sound ( S ) and second heart sound ( S ) , produced by the closing of the atrioventricular valves and semilunar valves , respectively
. In addition to these normal sounds , a variety of other sounds may be present including heart murmurs , adventitious sounds , and gallop rhythms S and S .
Heart murmurs are generated by turbulent flow of blood , which may occur inside or outside the heart .
</item 9>

<item 10>
Heart sounds
Aortic valve ( to aorta ) right second intercostal space upper right sternal border Pulmonary valve ( to pulmonary trunk ) left second intercostal space upper
left sternal border Erb 's point Left third intercostal space left sternal border Tricuspid valve ( to right ventricle ) left fourth , fifth intercostal spaces
lower left sternal border Mitral valve ( to left ventricle ) left fifth intercostal space left midclavicular line Recording heart sounds ( edit ) Using
electronic stethoscopes , it is possible to record heart sounds via direct output to an external recording device , such as a laptop or MP3 recorder . The same
connection can be used to listen to the previously recorded auscultation through the stethoscope headphones , allowing for more detailed study of murmurs and
other heart sounds , for general research as well as evaluation of a particular patient 's condition .
</item 10>
=== END ITEM id=31c5b9985739f731 ===
